import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# Import constants from your features module
from engine.features import FEATURES, PITCH_GROUPS

logger = logging.getLogger(__name__)

# ==========================================
# BIOMECHANICAL PRE-PROCESSING
# ==========================================

def preprocess_atlas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans incoming Statcast data and engineers biomechanical metrics.
    """
    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS)
    df = df.dropna(subset=['pfx_x', 'pfx_z', 'plate_x', 'plate_z'])
    
    # Kinematic Engineering
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_ratio'] = df['total_break'] / df['release_speed']
    df['reaction_time'] = (55 - df['release_extension']) / df['effective_speed']
    
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
        random_state=42
    )
    
    model.fit(X_train, y_train)
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
    
    # Apply weights to the raw dataset
    X_raw = df[FEATURES].fillna(0).copy()
    X_weighted = X_raw * normalized_weights
    
    # Train the Nearest Neighbors model on the dimensionally adjusted data
    nn_model = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_model.fit(X_weighted)
    
    logger.info("Weighted 1-NN model fitted successfully.")
    return nn_model, normalized_weights

def get_historical_clone(target_features: np.ndarray, nn_model: NearestNeighbors, weights: np.ndarray, original_df: pd.DataFrame):
    """
    Identifies the most mathematically similar historical pitch based on weighted Euclidean distance.
    """
    target_weighted = np.array(target_features) * weights
    distances, indices = nn_model.kneighbors([target_weighted])
    
    # indices[0][1] retrieves the nearest neighbor, excluding the target pitch itself
    clone_idx = indices[0][1]
    clone_pitch = original_df.iloc[clone_idx]
    
    return clone_pitch, distances[0][1]

def recommend_arsenal(
    target_features: np.ndarray,
    nn_model: NearestNeighbors,
    weights: np.ndarray,
    df: pd.DataFrame,
    pitcher_id_col: str = "pitcher",
    pitch_type_col: str = "pitch_type"
) -> dict:
    """
    Recommends a pitch arsenal based on the nearest historical pitch clone.
    
    Returns:
        {
            "clone_pitch": pd.Series,
            "distance": float,
            "arsenal": pd.DataFrame
        }
    """
    logger.info("Generating arsenal recommendation via 1-NN clone...")
    
    # Step 1: Find nearest neighbor (clone)
    clone_pitch, distance = get_historical_clone(
        target_features, nn_model, weights, df
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
            "arsenal": pd.DataFrame()
        }
    
    # Step 4: Build arsenal (pitch mix)
    arsenal = (
        pitcher_df[pitch_type_col]
        .value_counts(normalize=True)
        .reset_index()
    )
    
    arsenal.columns = ["pitch_type", "usage"]
    
    # Optional: map to pitch groups
    if "pitch_group" in pitcher_df.columns:
        group_arsenal = (
            pitcher_df["pitch_group"]
            .value_counts(normalize=True)
            .reset_index()
        )
        group_arsenal.columns = ["pitch_group", "usage"]
    else:
        group_arsenal = None
    
    logger.info(f"Arsenal generated for pitcher {clone_pitcher_id}")
    
    return {
        "clone_pitch": clone_pitch,
        "distance": distance,
        "arsenal": arsenal,
        "group_arsenal": group_arsenal
    }
