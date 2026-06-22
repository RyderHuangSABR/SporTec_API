import os
import numpy as np
import pandas as pd
import logging
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import hf_hub_download
from functools import lru_cache

# Import constants from your features module
from engine.features import FEATURES, PITCH_GROUPS

logger = logging.getLogger(__name__)

# ==========================================
# 1. LOAD THE REFERENCE DATA (THE DICTIONARY)
# ==========================================
@lru_cache(maxsize=None)
def load_reference_profiles() -> pd.DataFrame:
    """Downloads the pitch profiles so KNN knows who the row numbers belong to. Cached for performance."""
    token = os.getenv("HF_TOKEN")
    try:
        csv_path = hf_hub_download(
            repo_id="RyderHuangSABR/Atlas_Pitching_Data", 
            filename="Atlas/gmm_pitch_profiles.csv", 
            repo_type="dataset", 
            token=token
        )
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"HF Download Failed: {e}")
        raise FileNotFoundError("Could not pull reference CSV from Hugging Face.")

# ==========================================
# 2. LOAD JOBLIB MODELS (GMM & SCALER)
# ==========================================
@lru_cache(maxsize=None)
def load_sk_model(filename: str):
    """Dynamically downloads and loads a scikit-learn .joblib model. Cached for performance."""
    token = os.getenv("HF_TOKEN")
    try:
        logger.info(f"Pulling model: {filename}")
        model_path = hf_hub_download(
            repo_id="RyderHuangSABR/Atlas_Pitching_ML", 
            filename=f"{filename}", 
            token=token
        )
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        raise

# ==========================================
# 3. FEATURE WEIGHTING (FALLBACK)
# ==========================================
@lru_cache(maxsize=None)
def get_equal_weights() -> np.ndarray:
    """
    Returns an array of equal weights for all features.
    Replaces the XGBoost feature importance weighting.
    """
    # Defaulting to equal weights that sum to 1.0 to keep scaling stable
    return np.ones(len(FEATURES)) / len(FEATURES)

# ==========================================
# 4. MAIN RECOMMENDER 
# ==========================================
def recommend_arsenal(target_df: pd.DataFrame) -> dict:
    """Matches pitch using serialized GMM and KNN models."""
    logger.info("Initializing ML pipeline matching...")
    
    try:
        # 1. Load Reference Data and ML Models (Now lightning fast due to @lru_cache)
        profiles_df = load_reference_profiles()
        gmm_model = load_sk_model("gmm_baseline_2026.joblib")
        scaler = load_sk_model("scaler_baseline_2026.joblib")
        
        target_pitch_type = target_df['pitch_type'].iloc[0] if 'pitch_type' in target_df.columns else "FF"
        target_pitch_group = PITCH_GROUPS.get(target_pitch_type, "Fastball")
        
        # Identify the target pitcher's ID so we don't match them with themselves
        target_pitcher_id = None
        if 'pitcher' in target_df.columns:
            target_pitcher_id = target_df['pitcher'].iloc[0]
        elif 'MLBID' in target_df.columns:
            target_pitcher_id = target_df['MLBID'].iloc[0]
        
        # 2. Prepare Features
        for col in FEATURES:
            if col not in target_df.columns:
                target_df[col] = 0.0
            if col not in profiles_df.columns:
                profiles_df[col] = 0.0
                
        target_raw = target_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        candidates_raw = profiles_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 3. Standardize Data 
        target_scaled = scaler.transform(target_raw)
        candidates_scaled = scaler.transform(candidates_raw)
        
        # 4. GMM Prediction 
        cluster_id = gmm_model.predict(target_scaled)[0]
        logger.info(f"GMM assigned this pitch to Cluster ID: {cluster_id}")
        
        # 5. Apply Equal Weights (Replacing XGBoost Alpha)
        weights = get_equal_weights()
        target_weighted = target_scaled * weights
        candidates_weighted = candidates_scaled * weights
        
        # 6. Dynamic KNN (Search pool expanded to 15 to escape self-matching clusters)
        dynamic_knn = NearestNeighbors(n_neighbors=15)
        dynamic_knn.fit(candidates_weighted)
        distances, indices = dynamic_knn.kneighbors(target_weighted)
        
        best_idx = None
        best_dist = None
        
        # Loop through the top 15 closest pitches
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            candidate_mlbid = profiles_df.iloc[idx]['MLBID']
            
            # Check for the "Self-Match" bug: Skip if it's the exact same guy
            if target_pitcher_id is not None and pd.notna(target_pitcher_id):
                if int(candidate_mlbid) == int(target_pitcher_id):
                    continue 
            
            # If it's a different pitcher, lock it in and break the loop
            best_idx = idx
            best_dist = dist
            break
            
        # Fallback: If somehow all 15 matches were the exact same pitcher
        if best_idx is None:
            logger.warning(f"Pitcher {target_pitcher_id} consumed all 15 nearest neighbors. Defaulting to exact nearest.")
            best_idx = indices[0][0]
            best_dist = distances[0][0]
        
        # 7. Extract the Winning MLB Clone from our Reference CSV
        clone_pitch = profiles_df.iloc[best_idx].copy()
        clone_pitcher_id = clone_pitch['MLBID']
        matched_pitch_type = clone_pitch['pitch_type']
        
        # 8. Calculate Arsenal Usage
        pitcher_arsenal_df = profiles_df[profiles_df['MLBID'] == clone_pitcher_id].copy()
        total_pitches = pitcher_arsenal_df['sample_size'].sum()
        
        arsenal_data = []
        group_arsenal_dict = {}
        for _, row in pitcher_arsenal_df.iterrows():
            p_type = row['pitch_type']
            usage = row['sample_size'] / total_pitches
            arsenal_data.append({"pitch_type": p_type, "usage": float(usage)})
            
            p_group = PITCH_GROUPS.get(p_type, 'Unknown')
            group_arsenal_dict[p_group] = group_arsenal_dict.get(p_group, 0) + usage
            
        group_arsenal_data = [{"pitch_group": k, "usage": float(v)} for k, v in group_arsenal_dict.items()]
        arsenal_data = sorted(arsenal_data, key=lambda x: x['usage'], reverse=True)
        group_arsenal_data = sorted(group_arsenal_data, key=lambda x: x['usage'], reverse=True)
        
        # 9. Safety Net for Output
        # Convert any numpy types to native Python types for clean JSON serialization later
        clean_clone = {k: (None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v) for k, v in clone_pitch.items()}
        
        for col in target_df.columns:
            if col not in clean_clone:
                clean_clone[col] = target_df[col].iloc[0]

        # 10. Output
        logger.info(f"Matched pitcher {clone_pitcher_id} with a weighted distance of {best_dist:.4f}.")
        return {
            "identity": {
                "matched_pitcher_id": int(clone_pitcher_id),
                "matched_pitch_type": matched_pitch_type,
                "gmm_cluster_id": int(cluster_id),
                "coach_cue": f"This pitch moves and releases naturally like a {matched_pitch_type} from pitcher ID {int(clone_pitcher_id)}."
            },
            "arsenal": arsenal_data,
            "group_arsenal": group_arsenal_data,
            "distance": float(best_dist), 
            "clone_pitch": clean_clone    
        }
        
    except Exception as e:
        logger.error(f"Joblib Match Failed: {e}")
        return {
            "error": str(e), 
            "identity": None, 
            "arsenal": [], 
            "group_arsenal": [], 
            "distance": None,
            "clone_pitch": None
        }
