import os
import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download

_DB_CONNECTION = None

def get_db_connection():
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
        DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
        TARGET_FILE = "Atlas/cleaned_pitch_data.parquet" 
        
        try:
            file_path = hf_hub_download(repo_id=DATA_REPO, filename=TARGET_FILE, repo_type="dataset", token=HF_TOKEN)
            _DB_CONNECTION = duckdb.connect(database=':memory:')
            _DB_CONNECTION.execute(f"CREATE OR REPLACE VIEW mlb_history AS SELECT * FROM read_parquet('{file_path}')")
            print("💎 Hexcore Armed: Mathematical Vault Online.")
        except Exception as e:
            print(f"⚠️ Vault Bootstrap Failure: {e}")
            raise e
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    try:
        con = get_db_connection()
        
        # 1. Inputs
        t_vaa, t_haa = float(target_df['vaa'].iloc[0]), float(target_df['haa'].iloc[0])
        t_ext = float(target_df['release_extension'].iloc[0])
        t_z, t_x = float(target_df['release_pos_z'].iloc[0]), float(target_df['release_pos_x'].iloc[0])
        t_fb_speed, t_speed = float(target_df['fastball_speed'].iloc[0]), float(target_df['release_speed'].iloc[0])
        t_axis = float(target_df['spin_axis'].iloc[0])
        t_hand = str(target_df['p_throws'].iloc[0]).upper()
        t_velo_delta = t_fb_speed - t_speed 

        # 2. THE EVOLVED HEURISTIC: Weighted Euclidean Kinematics
        # We weigh VAA/HAA and Release Height higher (8.0-12.0) because they define the "tunnel"
        query = f"""
            SELECT 
                MLBID, pitch_type, Quality_Plus, spin_axis, release_speed, pfx_x, pfx_z,
                SQRT(
                    POWER((vaa - ({t_vaa})) * 12.0, 2) + 
                    POWER((haa - ({t_haa})) * 10.0, 2) + 
                    POWER((release_pos_z - ({t_z})) * 8.0, 2) + 
                    POWER((release_pos_x - ({t_x})) * 8.0, 2) + 
                    POWER((release_extension - ({t_ext})) * 4.0, 2) + 
                    POWER(((fb_speed - release_speed) - ({t_velo_delta})) * 5.0, 2) +
                    POWER(LEAST(ABS(spin_axis - ({t_axis})), 360 - ABS(spin_axis - ({t_axis}))) * 0.1, 2)
                ) as kinematic_distance
            FROM mlb_history
            WHERE pitch_type NOT IN ('FF', 'SI', 'FC') 
            AND p_throws = '{t_hand}'                  
            ORDER BY kinematic_distance ASC
            LIMIT 150                                  
        """
        
        clones = con.execute(query).df()
        if clones.empty: return {"status": "warning", "reason": "No physiological clones found."}

        # 3. THE "STABILITY" FILTER (Viktor's Logic: Reliability > Flashing Lights)
        # We calculate the CV (Coefficient of Variation) to avoid "wildcard" pitches
        stats = clones.groupby('pitch_type')['Quality_Plus'].agg(['mean', 'std', 'count']).fillna(0)
        stats = stats[stats['count'] > 5] # Must have at least 5 similar clones to be valid
        
        # Stability Score = Mean Quality / (Standard Deviation + 1)
        stats['stability'] = stats['mean'] / (stats['std'] + 1)
        best_pitch_type = stats['stability'].idxmax()
        
        # 4. Final Recommendation
        best_clone = clones[clones['pitch_type'] == best_pitch_type].sort_values('Quality_Plus', ascending=False).iloc[0]

        return {
            "status": "success",
            "recommended_pitch": str(best_clone['pitch_type']),
            "expected_quality_plus": round(float(best_clone['Quality_Plus']), 1),
            "optimal_spin_axis": round(float(best_clone['spin_axis']), 1),
            "optimal_pfx_x": round(float(best_clone['pfx_x']), 1),
            "optimal_pfx_z": round(float(best_clone['pfx_z']), 1),
            "apex_clone_mlbid": int(best_clone['MLBID']),
            "confidence_score": round(float(stats.loc[best_pitch_type, 'stability']), 2)
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}
