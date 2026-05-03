import os
import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Import constants from your features module
from engine.features import FEATURES, PITCH_GROUPS

logger = logging.getLogger(__name__)

# ==========================================
# 1. LOAD THE PRE-TRAINED GMM PROFILES
# ==========================================
def load_gmm_profiles() -> pd.DataFrame:
    """Loads the pre-calculated GMM centroids from the local CSV."""
    # This expects the Kaggle output to be saved in an 'atlas' folder in your project root
    csv_path = os.path.join("Atlas", "gmm_pitch_profiles.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Please place the Kaggle output in the 'atlas' folder.")
    
    return pd.read_csv(csv_path)

# ==========================================
# 2. MAIN RECOMMENDER (LIGHTNING FAST)
# ==========================================
def recommend_arsenal(target_df: pd.DataFrame) -> dict:
    """Matches the user's pitch to the closest pure GMM profile and returns coach-friendly data."""
    logger.info("Matching against GMM pitch profiles...")
    
    try:
        profiles_df = load_gmm_profiles()
        
        # 1. Prepare Target and Candidates
        for col in FEATURES:
            if col not in target_df.columns:
                target_df[col] = 0.0
                
        target_raw = target_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        candidates_raw = profiles_df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 2. Standardize Features (so Velo and Break are weighted equally)
        scaler = StandardScaler()
        scaler.fit(candidates_raw)
        
        target_weighted = scaler.transform(target_raw)
        candidates_weighted = scaler.transform(candidates_raw)
        
        # 3. K-Nearest Neighbors Math (Distance to GMM Centroids)
        distances = cdist(target_weighted, candidates_weighted, metric='euclidean')[0]
        best_idx = np.argsort(distances)[0]
        best_dist = distances[best_idx]
        
        # 4. Extract the Winning MLB Clone
        clone_pitch = profiles_df.iloc[best_idx].copy()
        clone_pitcher_id = clone_pitch['MLBID']
        matched_pitch_type = clone_pitch['pitch_type']
        
        # 5. Calculate Arsenal Usage directly from the GMM sample sizes
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

        # Sort arsenals by highest usage
        arsenal_data = sorted(arsenal_data, key=lambda x: x['usage'], reverse=True)
        group_arsenal_data = sorted(group_arsenal_data, key=lambda x: x['usage'], reverse=True)
        
        # 6. Safety Net for FastAPI Pydantic Models
        clean_clone = clone_pitch.replace({np.nan: None}).to_dict()
        for col in target_df.columns:
            if col not in clean_clone:
                clean_clone[col] = target_df[col].iloc[0]

        # 7. The Coach-Friendly Output JSON
        logger.info(f"Matched pitcher {clone_pitcher_id} in {best_dist:.2f} distance.")
        
        return {
            "identity": {
                "matched_pitcher_id": int(clone_pitcher_id),
                "matched_pitch_type": matched_pitch_type,
                "coach_cue": f"This pitch moves and releases naturally like a {matched_pitch_type} from pitcher ID {int(clone_pitcher_id)}."
            },
            "arsenal": arsenal_data,
            "group_arsenal": group_arsenal_data,
            "analytics": {
                "distance": float(best_dist),
                "clone_pitch": clean_clone
            }
        }
        
    except Exception as e:
        logger.error(f"GMM Match Failed: {e}")
        return {
            "error": str(e), 
            "identity": None, 
            "arsenal": [], 
            "group_arsenal": [], 
            "analytics": None
        }
