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
    original_df: pd.DataFrame,
    target_z: float, # Passing the raw Z coordinate
    target_x: float, # Passing the raw X coordinate
    z_tolerance: float = 0.33, # ~4 inches vertically
    x_tolerance: float = 0.50, # ~6 inches horizontally (x fluctuates more naturally)
    k_search: int = 50 # Search depth
):
    """
    Identifies the most mathematically similar historical pitch that ALSO 
    fits the pitcher's biomechanical arm slot constraints.
    """
    # 1. Prepare the target
    target_reshaped = np.array(target_features).reshape(1, -1)
    target_scaled = scaler.transform(target_reshaped)
    target_weighted = target_scaled * weights
    
    # 2. Retrieve a wider pool of nearest neighbors (top K instead of top 2)
    distances, indices = nn_model.kneighbors(target_weighted, n_neighbors=k_search)
    
    # 3. Iterate through neighbors to find the first biomechanical match
    for i in range(k_search):
        idx = indices[0][i]
        dist = distances[0][i]
        
        # Skip self-match (if the pitch is already in the dataset)
        if dist < 1e-6:
            continue
            
        candidate_pitch = original_df.iloc[idx]
        
        # Calculate arm slot delta
        z_diff = abs(candidate_pitch['release_pos_z'] - target_z)
        x_diff = abs(candidate_pitch['release_pos_x'] - target_x)
        
        # If it matches the arm slot criteria, we found our true clone!
        if z_diff <= z_tolerance and x_diff <= x_tolerance:
            return candidate_pitch, dist

    # 4. Fallback: If no pitch in the top 50 matches the slot, default to the closest non-self neighbor
    logger.warning("No clone found within arm slot tolerance. Falling back to absolute nearest neighbor.")
    fallback_idx = indices[0][1] if distances[0][0] < 1e-6 else indices[0][0]
    fallback_dist = distances[0][1] if distances[0][0] < 1e-6 else distances[0][0]
    
    return original_df.iloc[fallback_idx], fallback_dist

def recommend_arsenal(
    target_features: np.ndarray,
    target_dict: dict, # Pass the raw input dictionary to extract z and x
    nn_model: NearestNeighbors,
    scaler: StandardScaler,
    weights: np.ndarray,
    df: pd.DataFrame,
    pitcher_id_col: str = "pitcher",
    pitch_type_col: str = "pitch_type"
) -> dict:
    """
    Recommends a pitch arsenal based on the nearest historical pitch clone,
    constrained by arm slot.
    """
    logger.info("Generating biomechanically constrained arsenal recommendation...")
    
    # Extract the raw release coordinates from the incoming pitch
    target_z = target_dict.get('release_pos_z', 6.0) # Default to 6.0 if missing
    target_x = target_dict.get('release_pos_x', 2.0)
    
    # Step 1: Find nearest neighbor (clone) with arm slot filtering
    clone_pitch, distance = get_historical_clone(
        target_features, nn_model, scaler, weights, df, 
        target_z=target_z, target_x=target_x
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
        
    logger.info(f"Arsenal generated matching arm slot for pitcher {clone_pitcher_id}")
    
    return {
        "clone_pitch": clone_pitch,
        "distance": distance,
        "arsenal": arsenal,
        "group_arsenal": group_arsenal
    }
