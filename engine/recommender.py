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
            file_path = hf_hub_download(
                repo_id=DATA_REPO, 
                filename=TARGET_FILE, 
                repo_type="dataset", 
                token=HF_TOKEN
            )
            _DB_CONNECTION = duckdb.connect(database=':memory:')
            _DB_CONNECTION.execute(f"CREATE OR REPLACE VIEW mlb_history AS SELECT * FROM read_parquet('{file_path}')")
            print("💎 Hexcore Armed: Data loaded into memory.")
        except Exception as e:
            print(f"⚠️ Hexcore Boot Error: {e}")
            raise e
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    try:
        con = get_db_connection()
        
        # Extract Inputs
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

        # THE MATH: Weighted Euclidean Distance (Hexcore Optimization)
        query = f"""
            SELECT 
                MLBID, pitch_type, Quality_Plus, spin_axis, release_speed, pfx_x, pfx_z,
                SQRT(
                    POWER((vaa - ({t_vaa})) * 10.0, 2) + 
                    POWER((haa - ({t_haa})) * 10.0, 2) + 
                    POWER((release_extension - ({t_ext})) * 5.0, 2) + 
                    POWER((release_pos_z - ({t_z})) * 8.0, 2) + 
                    POWER((release_pos_x - ({t_x})) * 8.0, 2) + 
                    POWER(((fb_speed - release_speed) - ({t_velo_delta})) * 4.0, 2) +
                    POWER(LEAST(ABS(spin_axis - ({t_axis})), 360 - ABS(spin_axis - ({t_axis}))) * 0.1, 2)
                ) as k_dist
            FROM mlb_history
            WHERE pitch_type NOT IN ('FF', 'SI', 'FC') 
            AND p_throws = '{t_hand}'
            ORDER BY k_dist ASC
            LIMIT 75
        """
        clones = con.execute(query).df()

        if clones.empty:
            return {"status": "warning", "reason": "No physiological matches found."}

        # THE EVOLUTION: Pseudo-Sharpe Ratio (Stability Filter)
        # We group by pitch type and find which one is most consistently high-performing
        stats = clones.groupby('pitch_type')['Quality_Plus'].agg(['mean', 'std']).fillna(0)
        stats['stability'] = stats['mean'] / (stats['std'] + 1)
        best_type = stats['stability'].idxmax()
        
        # Pick the absolute king of that cluster
        best_clone = clones[clones['pitch_type'] == best_type].sort_values(by='Quality_Plus', ascending=False).iloc[0]

        return {
            "status": "success",
            "recommended_pitch": str(best_clone['pitch_type']),
            "expected_quality_plus": round(float(best_clone['Quality_Plus']), 1),
            "optimal_pfx_x": round(float(best_clone['pfx_x']), 1),
            "optimal_pfx_z": round(float(best_clone['pfx_z']), 1),
            "apex_clone_mlbid": int(best_clone['MLBID']),
            "reason": f"Stability filtered for {best_type}. Highest floor/ceiling ratio for your arm slot."
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}
