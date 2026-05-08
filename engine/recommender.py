import os
import numpy as np
import pandas as pd
import logging
import joblib 
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download

# Import constants from your features module
from engine.features import FEATURES, PITCH_GROUPS

logger = logging.getLogger(__name__)

# ==========================================
# 1. LOAD THE REFERENCE DATA (THE DICTIONARY)
# ==========================================
def load_reference_profiles() -> pd.DataFrame:
    """Downloads the pitch profiles so KNN knows who the row numbers belong to."""
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
# 2. LOAD JOBLIB MODELS (GMM & KNN)
# ==========================================
def load_sk_model(filename: str):
    """Dynamically downloads and loads a scikit-learn .joblib model."""
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
# 3. LOAD XGBOOST WEIGHTS (THE ALPHA)
# ==========================================
def load_xgb_weights(pitch_group: str) -> np.ndarray:
    """
    Extracts the feature importance (gain) from the pre-trained XGBoost Whiff model.
    These weights are used to penalize/reward the geometric distance in the KNN.
    """
    token = os.getenv("HF_TOKEN")
    filename = f"Engine_A_Whiff_{pitch_group}.json"
    filename = f"Engine_B_Contact_{pitch_group}.json"
    try:
        logger.info(f"Pulling XGBoost model: {filename}")
        model_path = hf_hub_download(
            repo_id="RyderHuangSABR/Atlas_Pitching_ML", 
            filename=filename, 
            token=token
        )
        
        # Load the Booster
        bst = xgb.Booster()
        bst.load_model(model_path)
        
        # Get feature importance based on 'gain'
        importances = bst.get_score(importance_type='gain')
        
        # Map the importances to our specific feature array, defaulting to 0.01 if missing
        weights = [importances.get(col, 0.01) for col in FEATURES]
        weights_array = np.array(weights)
        
        # Normalize the weights so they sum to 1.0 to keep scaling stable
        return weights_array / np.sum(weights_array)
        
    except Exception as e:
        logger.warning(f"XGB Weighting Failed for {filename}. Defaulting to equal weights. Error: {e}")
        return np.ones(len(FEATURES)) / len(FEATURES)

# ==========================================
# 4. MAIN RECOMMENDER 
# ==========================================
def recommend_arsenal(target_df: pd.DataFrame) -> dict:
    """Matches pitch using serialized GMM and Weighted KNN models."""
    logger.info("Initializing ML pipeline matching...")
    
    try:
        # 1. Load Reference Data and ML Models
        profiles_df = load_reference_profiles()
        gmm_model = load_sk_model("gmm_baseline_2026.joblib")
        scaler = load_sk_model("scaler_baseline_2026.joblib")
        knn_model = load_sk_model("knn_baseline_2026.joblib")
        
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
        
        # 3. Standardize & Weight Target Pitch (Applying XGBoost Alpha)
        scaler.fit(candidates_raw) # Note: If you have a pre-fit scaler, you might want to use it directly instead of re-fitting
        xgb_weights = load_xgb_weights(target_pitch_group)
        
        # Multiply standardized features by XGBoost feature importance
        target_weighted = scaler.transform(target_raw) * xgb_weights
        
        # 4. GMM Prediction
        cluster_id = gmm_model.predict(target_weighted)[0]
        logger.info(f"GMM assigned this pitch to Cluster ID: {cluster_id}")
        
        # 5. KNN Math (Ask for Top 5 to avoid Target Leakage)
        distances, indices = knn_model.kneighbors(target_weighted, n_neighbors=5)
        
        best_idx = None
        best_dist = None
        
        # Loop through the top 5 closest pitches
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            candidate_mlbid = profiles_df.iloc[idx]['MLBID']
            
            # Check for the "Self-Match" bug: Skip if it's the exact same guy
            if target_pitcher_id is not None and pd.notna(target_pitcher_id):
                if int(candidate_mlbid) == int(target_pitcher_id):
                    continue # Skip to the next closest neighbor
            
            # If it's a different pitcher, lock it in and break the loop
            best_idx = idx
            best_dist = dist
            break
            
        # Fallback: If somehow all 5 matches were the exact same pitcher
        if best_idx is None:
            best_idx = indices[0][1]
            best_dist = distances[0][1]
        
        # 6. Extract the Winning MLB Clone from our Reference CSV
        clone_pitch = profiles_df.iloc[best_idx].copy()
        clone_pitcher_id = clone_pitch['MLBID']
        matched_pitch_type = clone_pitch['pitch_type']
        
        # 7. Calculate Arsenal Usage
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
        
        # 8. Safety Net for Output
        clean_clone = clone_pitch.replace({np.nan: None}).to_dict()
        for col in target_df.columns:
            if col not in clean_clone:
                clean_clone[col] = target_df[col].iloc[0]

        # 9. Output
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
