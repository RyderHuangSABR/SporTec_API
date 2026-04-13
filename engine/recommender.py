import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
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
    # Drop NAs first, then use .copy() to avoid SettingWithCopyWarning
    clean_cols = ['pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_speed', 'effective_speed']
    df = df.dropna(subset=clean_cols).copy()
    
    # Safely map groups
    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS).fillna('Unknown')
    
    # Kinematic Engineering
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_ratio'] = df['total_break'] / df['release_speed']
    
    # Prevent division by zero if effective_speed is missing or corrupted
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
        early_stopping_rounds=50, # Prevents overfitting
        random_state=42
    )
    
    # Passing eval_set to track early stopping
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    logger.info(f"XGBoost training complete. Validation Damage RMSE: {rmse:.3f}")
    
    return model

# ==========================================
# WEIGHTED 1-NN CLONE RETRIEVAL
# ==========================================

def build_weighted_knn(xgb_model: xgb.XGBRegressor, df: pd.DataFrame):
    """
    Constructs a 1-NN recommender using XGBoost feature importances 
    to dimensionality-weight the feature space.
    """
    logger.info("Constructing Weighted 1-NN model...")
    
    # Extract and normalize XGBoost feature importances
    xgb_weights = xgb_model.feature_importances_
    normalized_weights = xgb_weights / np.sum(xgb_weights)
    
    X_raw = df[FEATURES].fillna(0).copy()
    
    # CRITICAL FIX: Scale the data before applying weights. 
    # Otherwise, features with large scales (spin rate) ruin the Euclidean distance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Apply importance weights to the normalized dataset
    X_weighted = X_scaled * normalized_weights
    
    # Train the Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_model.fit(X_weighted)
    
    logger.info("Weighted 1-NN model fitted successfully.")
    return nn_model, scaler, normalized_weights

def get_historical_clone(
    target_features: np.ndarray, 
    nn_model: NearestNeighbors, 
    scaler: StandardScaler, 
    weights: np.ndarray, 
    original_df: pd.DataFrame
):
    """
    Identifies the most mathematically similar historical pitch based on weighted Euclidean distance.
    """
    # Ensure 2D shape, scale, and weight the target features
    target_reshaped = np.array(target_features).reshape(1, -1)
    target_scaled = scaler.transform(target_reshaped)
    target_weighted = target_scaled * weights
    
    distances, indices = nn_model.kneighbors(target_weighted)
    
    # CRITICAL FIX: If distance to index 0 is ~0, the pitch is already in the dataset. 
    # If the distance > 0, it's a new pitch, and index 0 is the actual clone.
    if distances[0][0] < 1e-6:
        clone_idx = indices[0][1]
        clone_dist = distances[0][1]
    else:
        clone_idx = indices[0][0]
        clone_dist = distances[0][0]
        
    clone_pitch = original_df.iloc[clone_idx]
    
    return clone_pitch, clone_dist

def recommend_arsenal(
    target_features: np.ndarray,
    nn_model: NearestNeighbors,
    scaler: StandardScaler,
    weights: np.ndarray,
    df: pd.DataFrame,
    pitcher_id_col: str = "pitcher",
    pitch_type_col: str = "pitch_type"
) -> dict:
    """
    Recommends a pitch arsenal based on the nearest historical pitch clone.
    """
    logger.info("Generating arsenal recommendation via 1-NN clone...")
    
    # Step 1: Find nearest neighbor (clone)
    clone_pitch, distance = get_historical_clone(
        target_features, nn_model, scaler, weights, df
    )
    
    # Step 2: Identify the pitcher of the clone
    clone_pitcher_id = clone_pitch[pitcher_id_col]
    
    # Step 3: Get all pitches from that pitcher
    pitcher_df = df[df[pitcher_id_col] == clone_pitcher_id].copy()
    
    if pitcher_df.empty:
        logger.warning("No pitches found for clone pitcher.")
        return {
            "clone_pitch": clone_pitch,
            "distance": distance,
            "arsenal": pd.DataFrame(),
            "group_arsenal": None
        }
    
    # Step 4: Build arsenal (pitch mix)
    arsenal = pitcher_df[pitch_type_col].value_counts(normalize=True).reset_index()
    arsenal.columns = ["pitch_type", "usage"]
    
    # Optional: map to pitch groups
    group_arsenal = None
    if "pitch_group" in pitcher_df.columns:
        group_arsenal = pitcher_df["pitch_group"].value_counts(normalize=True).reset_index()
        group_arsenal.columns = ["pitch_group", "usage"]
    
    logger.info(f"Arsenal generated for pitcher {clone_pitcher_id}")
    
    return {
        "clone_pitch": clone_pitch,
        "distance": distance,
        "arsenal": arsenal,
        "group_arsenal": group_arsenal
    }
