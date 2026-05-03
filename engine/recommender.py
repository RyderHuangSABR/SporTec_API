import os
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import duckdb
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download

# Import constants from your features module
from engine.features import FEATURES, PITCH_GROUPS

logger = logging.getLogger(__name__)

# ==========================================
# FILE MANAGEMENT & DB CONNECTION
# ==========================================

def get_parquet_path() -> str:
    """Safely gets the path to the Parquet file on disk."""
    token = os.getenv("HF_TOKEN")
    return hf_hub_download(
        repo_id="RyderHuangSABR/Atlas_Pitching_Data", 
        filename="Atlas/Atlas_Pitching.parquet", 
        repo_type="dataset", 
        token=token
    )

def get_duckdb_conn():
    """Creates a strictly memory-leashed DuckDB connection."""
    con = duckdb.connect()
    # STRICT LEASH: Limit to 256MB RAM and 1 CPU thread
    con.execute("PRAGMA memory_limit='256MB'")
    con.execute("PRAGMA threads=1")
    return con

# ==========================================
# BIOMECHANICAL PRE-PROCESSING
# ==========================================

def preprocess_atlas_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans incoming Statcast data and engineers biomechanical metrics."""
    clean_cols = ['pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_speed', 'effective_speed']
    df = df.dropna(subset=clean_cols).copy()
    
    if df.empty:
        return df

    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS).fillna('Unknown')
    
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_ratio'] = df['total_break'] / df['release_speed']
    
    safe_speed = np.where(df['effective_speed'] == 0, 1e-5, df['effective_speed'])
    df['reaction_time'] = (55 - df['release_extension']) / safe_speed
    
    return df

# ==========================================
# DUCKDB LOW-RAM DISTANCE METRICS
# ==========================================

def get_strict_biomechanical_clone(
    target_df: pd.DataFrame, 
    z_tolerance: float = 0.25,
    x_tolerance: float = 0.33
):
    """Forces an absolute arm slot match natively from disk."""
    target_z = float(target_df['release_pos_z'].iloc[0])
    target_x = float(target_df['release_pos_x'].iloc[0])

    parquet_file = get_parquet_path()
    con = get_duckdb_conn()

    # 1. The Strict Arm Slot Gate (Leashed DuckDB Magic + RAM Safe Limit)
    query = f"""
        SELECT * FROM '{parquet_file}'
        WHERE release_pos_z BETWEEN {target_z - z_tolerance} AND {target_z + z_tolerance}
          AND release_pos_x BETWEEN {target_x - x_tolerance} AND {target_x + x_tolerance}
        LIMIT 5000
    """
    try:
        slot_df = con.query(query).df()
    finally:
        con.close()

    if slot_df.empty:
        raise ValueError(f"No historical pitches found within {z_tolerance}ft Z and {x_tolerance}ft X.")

    slot_df = preprocess_atlas_data(slot_df)

    scaler = StandardScaler()
    
    for col in FEATURES:
        if col not in target_df.columns:
            target_df[col] = 0.0
        if col not in slot_df.columns:
            slot_df[col] = 0.0

    target_raw = target_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    candidates_raw = slot_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    scaler.fit(candidates_raw)

    weights = np.ones(len(FEATURES))
    
    target_weighted = scaler.transform(target_raw) * weights
    candidates_weighted = scaler.transform(candidates_raw) * weights

    distances = cdist(target_weighted, candidates_weighted, metric='euclidean')[0]

    sorted_indices = np.argsort(distances)
    best_idx = sorted_indices[0]
    best_dist = distances[best_idx]

    clone_pitch = slot_df.iloc[best_idx]

    return clone_pitch, best_dist

# ==========================================
# MAIN RECOMMENDER
# ==========================================

def recommend_arsenal(target_df: pd.DataFrame, pitcher_id_col: str = "pitcher", pitch_type_col: str = "pitch_type") -> dict:
    """Recommends a pitch arsenal based on the strict biomechanical clone."""
    logger.info("Generating strictly constrained arsenal recommendation...")
    
    try:
        clone_pitch, distance = get_strict_biomechanical_clone(target_df)
    except Exception as e:
        logger.error(f"Arsenal Recommendation Failed: {e}")
        return {
            "error": str(e),
            "clone_pitch": {"pitcher_id": None},
            "distance": None,
            "arsenal": [],
            "group_arsenal": []
        }
    
    # 🚨 FIX 1: Safely extract Pitcher ID (handles floats, strings, and missing data)
    raw_id = clone_pitch.get(pitcher_id_col)
    if pd.isna(raw_id) and 'pitcher_id' in clone_pitch:
        raw_id = clone_pitch.get('pitcher_id')
        
    try:
        clone_pitcher_id = int(float(raw_id))
    except (ValueError, TypeError):
        return
