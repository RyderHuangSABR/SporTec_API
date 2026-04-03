import os
import pandas as pd
import duckdb
import xgboost as xgb
from huggingface_hub import hf_hub_download

# =====================================================================
# 🧠 ATLAS ENGINE HYPERPARAMETERS & CONFIG
# =====================================================================
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
ML_REPO = "RyderHuangSABR/Atlas_Pitching_ML"

PITCH_TYPES = ["Fastball", "Sinker", "Cutter", "SplitterFork", "Curveball", "Changeup", "SweeperSlider"]

XGB_WEIGHTS = {
    "vaa": 10.0, "haa": 10.0, "extension": 5.0, 
    "pos_z": 8.0, "pos_x": 8.0, "velo_delta": 3.0, "spin_axis": 0.1
}

CHASSIS_TOLERANCE_FT = 0.5 

# Global caches for instant API response times
_DB_CONNECTION = None
_XGB_MODELS = {}

# =====================================================================
# 🛠️ SYSTEM BOOTSTRAP (Loads into memory once)
# =====================================================================
def get_db_connection():
    """Bootstraps the DuckDB 1-NN Historical Vault."""
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        print("⚙️ Fetching DuckDB Vault from HuggingFace...")
        file_path = hf_hub_download(repo_id=DATA_REPO, filename="Atlas/cleaned_pitch_data.parquet", repo_type="dataset", token=HF_TOKEN)
        _DB_CONNECTION = duckdb.connect(database=':memory:')
        _DB_CONNECTION.execute(f"CREATE OR REPLACE VIEW mlb_history AS SELECT * FROM read_parquet('{file_path}')")
        print("💎 Historical Vault Armed.")
    return _DB_CONNECTION

def get_xgb_models():
    """Fetches and caches all 14 XGBoost JSON models from HuggingFace."""
    global _XGB_MODELS
    if not _XGB_MODELS:
        print("⚙️ Fetching 14 XGBoost Simulation Engines...")
        for pitch in PITCH_TYPES:
            _XGB_MODELS[pitch] = {}
            for engine, prefix in [("A_Whiff", "Engine_A_Whiff"), ("B_Contact", "Engine_B_Contact")]:
                filename = f"{prefix}_{pitch}.json"
                file_path = hf_hub_download(repo_id=ML_REPO, filename=filename, repo_type="model", token=HF_TOKEN)
                
                # Load XGBoost model from JSON
                model = xgb.Booster()
                model.load_model(file_path)
                _XGB_MODELS[pitch][engine] = model
        print("🔥 XGBoost Simulation Engines Armed.")
    return _XGB_MODELS

# =====================================================================
# 🚀 THE ORACLE (The Core API Endpoint)
# =====================================================================
def recommend_arsenal(target_df):
    """
    Two-Tier Oracle: 
    1. Simulates pitch quality via 14 XGBoost models.
    2. Validates physical reality via DuckDB 1-NN search.
    """
    try:
        con = get_db_connection()
        models = get_xgb_models()
        
        # --- STEP 1: THE SIMULATION (XGBoost) ---
        dmatrix_input = xgb.DMatrix(target_df)
        pitch_scores = []
        
        for pitch in PITCH_TYPES:
            # Predict Whiff (Higher is better) and Contact (Depends on your scale, assuming higher is better/safer here)
            whiff_score = models[pitch]["A_Whiff"].predict(dmatrix_input)[0]
            contact_score = models[pitch]["B_Contact"].predict(dmatrix_input)[0]
            
            # Blend into a master Quality+ Metric (Adjust math based on your model's exact scaling)
            composite_score = (whiff_score * 0.6) + (contact_score * 0.4) 
            pitch_scores.append({"pitch_type": pitch, "score": composite_score})
            
        # Rank pitches mathematically
        pitch_scores = sorted(pitch_scores, key=lambda x: x["score"], reverse=True)
        best_pitch_simulated = pitch_scores[0]["pitch_type"]
        best_pitch_score = pitch_scores[0]["score"]

        # --- STEP 2: THE REALITY CHECK (DuckDB 1-NN) ---
        # Extract kinematics to find the twin
        t_vaa = float(target_df['vaa'].iloc[0])
        t_haa = float(target_df['haa'].iloc[0])
        t_ext = float(target_df['release_extension'].iloc[0])
        t_z = float(target_df['release_pos_z'].iloc[0])
        t_x = float(target_df['release_pos_x'].iloc[0])
        t_fb_speed = float(target_df['fastball_speed'].iloc[0]) 
        t_speed = float(target_df['release_speed'].iloc[0])
        t_velo_delta = t_fb_speed - t_speed 
        t_axis = float(target_df['spin_axis'].iloc[0])
        t_hand = str(target_df['p_throws'].iloc[0]).upper()

        # NOTE: You may need a dictionary here to map your ML pitch names to Statcast pitch codes (e.g., "Changeup" -> "CH")
        statcast_pitch_code_map = {
            "Changeup": "CH", "SweeperSlider": "ST", "Curveball": "CU", 
            "SplitterFork": "FS", "Fastball": "FF", "Sinker": "SI", "Cutter": "FC"
        }
        target_pitch_code = statcast_pitch_code_map.get(best_pitch_simulated, best_pitch_simulated)

        query = f"""
            SELECT 
                MLBID, 
                pitch_type, 
                spin_axis,
                pfx_x,
                pfx_z,
                SQRT(
                    POWER((vaa - ({t_vaa})) * {XGB_WEIGHTS['vaa']}, 2) + 
                    POWER((haa - ({t_haa})) * {XGB_WEIGHTS['haa']}, 2) + 
                    POWER((release_extension - ({t_ext})) * {XGB_WEIGHTS['extension']}, 2) + 
                    POWER((release_pos_z - ({t_z})) * {XGB_WEIGHTS['pos_z']}, 2) + 
                    POWER((release_pos_x - ({t_x})) * {XGB_WEIGHTS['pos_x']}, 2) + 
                    POWER(((fastball_speed - release_speed) - ({t_velo_delta})) * {XGB_WEIGHTS['velo_delta']}, 2) +
                    POWER(LEAST(ABS(spin_axis - ({t_axis})), 360 - ABS(spin_axis - ({t_axis}))) * {XGB_WEIGHTS['spin_axis']}, 2)
                ) as kinematic_distance
            FROM mlb_history
            WHERE pitch_type = '{target_pitch_code}' 
            AND p_throws = '{t_hand}'
            AND release_pos_z BETWEEN ({t_z} - {CHASSIS_TOLERANCE_FT}) AND ({t_z} + {CHASSIS_TOLERANCE_FT})
            AND release_pos_x BETWEEN ({t_x} - {CHASSIS_TOLERANCE_FT}) AND ({t_x} + {CHASSIS_TOLERANCE_FT})
            AND release_extension BETWEEN ({t_ext} - {CHASSIS_TOLERANCE_FT}) AND ({t_ext} + {CHASSIS_TOLERANCE_FT})
            ORDER BY kinematic_distance ASC
            LIMIT 1
        """
        
        clone = con.execute(query).df()

        if clone.empty:
            return {
                "status": "warning",
                "simulated_pitch": best_pitch_simulated,
                "reason": f"XGBoost optimized for a {best_pitch_simulated}, but 1-NN found NO historical twin within safe physical tolerances. Pitch rejected as injury risk."
            }

        # --- STEP 3: RETURN THE ORACLE RESULT ---
        best_clone = clone.iloc[0]
        return {
            "status": "success",
            "recommended_pitch": best_pitch_simulated,
            "simulated_composite_score": round(float(best_pitch_score), 3),
            "historical_validation": {
                "twin_mlbid": int(best_clone['MLBID']),
                "twin_spin_axis": round(float(best_clone['spin_axis']), 1),
                "twin_pfx_x": round(float(best_clone['pfx_x']), 1),
                "twin_pfx_z": round(float(best_clone['pfx_z']), 1),
                "kinematic_distance": round(float(best_clone['kinematic_distance']), 3)
            },
            "reason": f"Engine Simulation chose {best_pitch_simulated}. 1-NN Vault validated physical reality via MLBID {int(best_clone['MLBID'])}."
        }
        
    except Exception as e:
        print(f"Recommender Error: {e}")
        return {"status": "error", "recommended_pitch": "Engine Fault", "reason": str(e)}

if __name__ == "__main__":
    # Test Payload
    sample_payload = pd.DataFrame([{
        "vaa": -5.2, "haa": 1.8, "release_extension": 6.8, "release_pos_z": 5.8,
        "release_pos_x": -2.1, "fastball_speed": 94.5, "release_speed": 85.2,
        "spin_axis": 210, "p_throws": "R"
    }])
    
    # Needs HF_TOKEN exported in environment or code to run models
    result = recommend_arsenal(sample_payload)
    print(result)
