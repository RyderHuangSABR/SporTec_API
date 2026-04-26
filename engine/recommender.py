import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import constants from your features module
from engine.loader import get_models_for_pitch
from engine.scoring import score_pitch
from engine.features import FEATURES, PITCH_GROUPS

logger = logging.getLogger(__name__)

# ==========================================
# BIOMECHANICAL PRE-PROCESSING
# ==========================================

def preprocess_atlas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans incoming Statcast data and engineers biomechanical metrics.
    """
    clean_cols = ['pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_speed', 'effective_speed']
    df = df.dropna(subset=clean_cols).copy()
    
    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS).fillna('Unknown')
    
    # Kinematic Engineering
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_ratio'] = df['total_break'] / df['release_speed']
    
    safe_speed = np.where(df['effective_speed'] == 0, 1e-5, df['effective_speed'])
    df['reaction_time'] = (55 - df['release_extension']) / safe_speed
    
    return df

# ==========================================
# XGBOOST CORE
# ==========================================

def train_xgboost_model(df: pd.DataFrame, target_col: str = 'contact_damage') -> xgb.XGBRegressor:
    """
    Trains the XGBoost regressor to predict Contact Damage RMSE.
    """
    logger.info("Initializing XGBoost training sequence...")
    
    X = df[FEATURES].fillna(0)
    y = df[target_col].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500, 
        learning_rate=0.05,
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    logger.info(f"XGBoost training complete. Validation Damage RMSE: {rmse:.3f}")
    
    return model

# ==========================================
# WEIGHTED DISTANCE METRICS
# ==========================================

def prepare_distance_metrics(xgb_model: xgb.XGBRegressor, df: pd.DataFrame):
    """
    Extracts weights and fits the scaler. 
    (Replaces the old KD-Tree approach for faster dynamic filtering)
    """
    logger.info("Preparing Scaler and XGBoost Weights for distance calculations...")
    
    xgb_weights = xgb_model.feature_importances_
    normalized_weights = xgb_weights / np.sum(xgb_weights)
    
    X_raw = df[FEATURES].fillna(0).copy()
    
    scaler = StandardScaler()
    scaler.fit(X_raw) # We only need to fit it, not transform the whole dataset yet
    
    logger.info("Metrics prepared successfully.")
    return scaler, normalized_weights

def get_strict_biomechanical_clone(
    target_features: np.ndarray, 
    target_dict: dict,
    scaler: StandardScaler, 
    weights: np.ndarray, 
    original_df: pd.DataFrame,
    z_tolerance: float = 0.25, # Strict +/- 3 inches vertically
    x_tolerance: float = 0.33  # Strict +/- 4 inches horizontally
):
    """
    Forces an absolute arm slot match by pre-filtering the dataset before 
    calculating the weighted Euclidean distance using cdist.
    """
    target_z = target_dict.get('release_pos_z', 6.0)
    target_x = target_dict.get('release_pos_x', 2.0)

    # 1. The Strict Arm Slot Gate
    slot_df = original_df[
        (original_df['release_pos_z'].between(target_z - z_tolerance, target_z + z_tolerance)) &
        (original_df['release_pos_x'].between(target_x - x_tolerance, target_x + x_tolerance))
    ].copy()

    if slot_df.empty:
        raise ValueError(f"No historical pitches found within {z_tolerance}ft Z and {x_tolerance}ft X.")

    # 2. Prepare the target vector
    target_reshaped = np.array(target_features).reshape(1, -1)
    target_weighted = scaler.transform(target_reshaped) * weights

    # 3. Prepare the filtered candidate pool
    candidates_raw = slot_df[FEATURES].fillna(0)
    candidates_weighted = scaler.transform(candidates_raw) * weights

    # 4. Vectorized Distance Calculation (Lightning Fast)
    distances = cdist(target_weighted, candidates_weighted, metric='euclidean')[0]

    # 5. Sort and find the best match
    sorted_indices = np.argsort(distances)
    best_idx_in_slot = sorted_indices[0]
    best_dist = distances[best_idx_in_slot]

    # 6. Protect against self-matching
    if best_dist < 1e-6 and len(sorted_indices) > 1:
        best_idx_in_slot = sorted_indices[1]
        best_dist = distances[best_idx_in_slot]

    clone_pitch = slot_df.iloc[best_idx_in_slot]

    return clone_pitch, best_dist

def recommend_arsenal(
    target_features: np.ndarray,
    target_dict: dict,
    scaler: StandardScaler,
    weights: np.ndarray,
    df: pd.DataFrame,
    pitcher_id_col: str = "pitcher",
    pitch_type_col: str = "pitch_type"
) -> dict:
    """
    Recommends a pitch arsenal based on the strict biomechanical clone.
    """
    logger.info("Generating strictly constrained arsenal recommendation...")
    
    # Try to find a clone. If the arm slot is too weird, catch the error cleanly.
    try:
        clone_pitch, distance = get_strict_biomechanical_clone(
            target_features, target_dict, scaler, weights, df
        )
    except ValueError as e:
        logger.error(f"Arsenal Recommendation Failed: {e}")
        return {
            "error": str(e),
            "clone_pitch": None,
            "distance": None,
            "arsenal": pd.DataFrame(),
            "group_arsenal": None
        }
    
    clone_pitcher_id = clone_pitch[pitcher_id_col]
    pitcher_df = df[df[pitcher_id_col] == clone_pitcher_id].copy()
    
    if pitcher_df.empty:
        return {"clone_pitch": clone_pitch, "distance": distance, "arsenal": pd.DataFrame(), "group_arsenal": None}
    
    arsenal = pitcher_df[pitch_type_col].value_counts(normalize=True).reset_index()
    arsenal.columns = ["pitch_type", "usage"]
    
    group_arsenal = None
    if "pitch_group" in pitcher_df.columns:
        group_arsenal = pitcher_df["pitch_group"].value_counts(normalize=True).reset_index()
        group_arsenal.columns = ["pitch_group", "usage"]
        
    logger.info(f"Arsenal generated matching arm slot for pitcher {clone_pitcher_id}")
    
    return {
        "clone_pitch": clone_pitch,
        "distance": distance,
        "arsenal": arsenal,
        "group_arsenal": group_arsenal
    }
