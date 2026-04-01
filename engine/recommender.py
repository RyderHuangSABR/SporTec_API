import os
import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download
from supabase import create_client, Client

_DB_CONNECTION = None

def get_db_connection():
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
        DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
        TARGET_FILE = "Atlas/cleaned_pitch_data.parquet" 
        
        # Supabase Config for "The Evolution"
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        
        try:
            file_path = hf_hub_download(repo_id=DATA_REPO, filename=TARGET_FILE, repo_type="dataset", token=HF_TOKEN)
            _DB_CONNECTION = duckdb.connect(database=':memory:')
            
            # 1. Load the Historical Base (The Flesh)
            _DB_CONNECTION.execute(f"CREATE OR REPLACE VIEW mlb_history AS SELECT * FROM read_parquet('{file_path}')")
            
            # 2. Pull Live Feedback from Supabase (The Hexcore Evolution)
            if SUPABASE_URL and SUPABASE_KEY:
                sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                # Pull the last 5,000 "successful" predictions from other users
                live_data = sb.table("prediction_logs").select("*").limit(5000).execute()
                
                if live_data.data:
                    live_df = pd.DataFrame(live_data.data)
                    _DB_CONNECTION.register("live_evolution", live_df)
                    print("🧬 Hexcore integrated live feedback logs.")
                else:
                    # Create empty placeholder if no logs exist yet
                    _DB_CONNECTION.execute("CREATE TABLE live_evolution AS SELECT * FROM mlb_history WHERE 1=0")
            
            print("💎 Atlas Engine Armed and Ready.")
        except Exception as e:
            print(f"⚠️ Vault bootstrap failed: {e}")
            raise e
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    try:
        con = get_db_connection()
        
        # Extract features
        t_vaa, t_haa = float(target_df['vaa'].iloc[0]), float(target_df['haa'].iloc[0])
        t_ext = float(target_df['release_extension'].iloc[0])
        t_z, t_x = float(target_df['release_pos_z'].iloc[0]), float(target_df['release_pos_x'].iloc[0])
        t_fb_speed = float(target_df['fastball_speed'].iloc[0]) 
        t_speed = float(target_df['release_speed'].iloc[0])
        t_velo_delta = t_fb_speed - t_speed 
        t_axis = float(target_df['spin_axis'].iloc[0])
        t_hand = str(target_df['p_throws'].iloc[0]).upper()

        # THE EVOLVED QUERY: Search BOTH MLB History AND User Logs
        # We weight the "Live Evolution" slightly higher to reflect modern trends
        query = f"""
            WITH united_data AS (
                SELECT MLBID as source_id, pitch_type, Quality_Plus, vaa, haa, release_extension, release_pos_z, release_pos_x, (94.0 - release_speed) as fb_diff, spin_axis, p_throws FROM mlb_history
                UNION ALL
                SELECT 999999 as source_id, recommended_pitch as pitch_type, expected_quality_plus as Quality_Plus, vaa, haa, release_extension, release_pos_z, release_pos_x, (fastball_speed - release_speed) as fb_diff, spin_axis, p_throws FROM live_evolution
            )
            SELECT *,
                SQRT(
                    POWER((vaa - ({t_vaa})) * 12.0, 2) + 
                    POWER((haa - ({t_haa})) * 12.0, 2) + 
                    POWER((release_extension - ({t_ext})) * 5.0, 2) + 
                    POWER((release_pos_z - ({t_z})) * 10.0, 2) + 
                    POWER((release_pos_x - ({t_x})) * 10.0, 2) + 
                    POWER((fb_diff - ({t_velo_delta})) * 4.0, 2) +
                    POWER(LEAST(ABS(spin_axis - ({t_axis})), 360 - ABS(spin_axis - ({t_axis}))) * 0.15, 2)
                ) as kinematic_distance
            FROM united_data
            WHERE p_throws = '{t_hand}'
            ORDER BY kinematic_distance ASC
            LIMIT 150
        """
        
        clones = con.execute(query).df()
        
        # Stability logic (Hedge-Fund filter)
        stats = clones.groupby('pitch_type')['Quality_Plus'].agg(['mean', 'std']).fillna(0)
        stats['stability_score'] = stats['mean'] / (stats['std'] + 1)
        best_pitch_type = stats['stability_score'].idxmax()
        
        best_clone = clones[clones['pitch_type'] == best_pitch_type].sort_values(by='Quality_Plus', ascending=False).iloc[0]

        return {
            "status": "success",
            "recommended_pitch": str(best_clone['pitch_type']),
            "expected_quality_plus": round(float(best_clone['Quality_Plus']), 1),
            "optimal_spin_axis": round(float(best_clone['spin_axis']), 1),
            "optimal_pfx_x": 0.0, # Placeholder if not in united_data
            "optimal_pfx_z": 0.0,
            "reason": f"Evolved match based on modern chassis trends and {t_hand}HP arm slot."
        }
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}
