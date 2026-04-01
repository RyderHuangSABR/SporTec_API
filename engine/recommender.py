import os
import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download

# Global connection to keep DuckDB in-memory across API calls
_DB_CONNECTION = None

def get_db_connection():
    """Bootstraps the DuckDB in-memory vault on first load."""
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        # 1. Grab the secure token from Render's environment variables
        HF_TOKEN = os.getenv("HF_TOKEN")
        DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
        
        # 2. Point to the ACTUAL Atlas feature dataset
        TARGET_FILE = "Atlas/cleaned_pitch_data.parquet" 
        
        print(f"⚙️ Fetching {TARGET_FILE} from HuggingFace Vault...")
        try:
            # Pull the historical data into the container's temporary storage
            file_path = hf_hub_download(
                repo_id=DATA_REPO, 
                filename=TARGET_FILE, 
                repo_type="dataset", 
                token=HF_TOKEN
            )
            
            # Initialize DuckDB in-memory for zero-latency queries
            _DB_CONNECTION = duckdb.connect(database=':memory:')
            
            # BULLETPROOFING: Added OR REPLACE to prevent hot-reload crashes
            if TARGET_FILE.endswith('.parquet'):
                _DB_CONNECTION.execute(f"CREATE OR REPLACE VIEW mlb_history AS SELECT * FROM read_parquet('{file_path}')")
            else:
                _DB_CONNECTION.execute(f"CREATE OR REPLACE VIEW mlb_history AS SELECT * FROM read_csv_auto('{file_path}')")
                
            print("💎 Atlas Engine Armed and Ready.")
        except Exception as e:
            print(f"⚠️ Failed to bootstrap vault: {e}")
            raise e
            
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    """
    The Atlas OS Core: Weighted KNN via DuckDB SQL.
    Maps an incoming pitcher's biomechanical chassis to the exact MLB historical clone.
    """
    try:
        con = get_db_connection()
        
        # 1. Extract physical data and baseline parameters from payload
        t_vaa = float(target_df['vaa'].iloc[0])
        t_haa = float(target_df['haa'].iloc[0])
        t_ext = float(target_df['release_extension'].iloc[0])
        t_z = float(target_df['release_pos_z'].iloc[0])
        t_x = float(target_df['release_pos_x'].iloc[0])
        
        # Velocity Delta: How much slower is this pitch than their primary fastball?
        t_fb_speed = float(target_df['fastball_speed'].iloc[0]) 
        t_speed = float(target_df['release_speed'].iloc[0])
        t_velo_delta = t_fb_speed - t_speed 
        
        t_axis = float(target_df['spin_axis'].iloc[0])
        t_hand = str(target_df['p_throws'].iloc[0]).upper()

        # 2. THE HOLY GRAIL: Weighted Euclidean Kinematic Distance
        # BULLETPROOFING: Wrapped all injected variables in parentheses to prevent double-negative SQL errors.
        query = f"""
            SELECT 
                MLBID, 
                pitch_type, 
                Quality_Plus,
                spin_axis,
                release_speed,
                pfx_x,
                pfx_z,
                SQRT(
                    POWER((vaa - ({t_vaa})) * 10.0, 2) + 
                    POWER((haa - ({t_haa})) * 10.0, 2) + 
                    POWER((release_extension - ({t_ext})) * 5.0, 2) + 
                    POWER((release_pos_z - ({t_z})) * 8.0, 2) + 
                    POWER((release_pos_x - ({t_x})) * 8.0, 2) + 
                    POWER(((fb_speed - release_speed) - ({t_velo_delta})) * 3.0, 2) +
                    POWER(LEAST(ABS(spin_axis - ({t_axis})), 360 - ABS(spin_axis - ({t_axis}))) * 0.1, 2)
                ) as kinematic_distance
            FROM mlb_history
            WHERE pitch_type NOT IN ('FF', 'SI', 'FC') -- Filter out fastballs to isolate secondaries
            AND p_throws = '{t_hand}'                  -- Quarantine by handedness
            ORDER BY kinematic_distance ASC
            LIMIT 100                                  -- Extract the 100 closest physiological clones
        """
        
        clones = con.execute(query).df()

        if clones.empty:
            return {
                "status": "warning",
                "recommended_pitch": "Unique Profile", 
                "reason": f"No historical {t_hand}HP clones match these specific kinematics."
            }

        # 3. Hedge-Fund Logic: The Stability Filter
        stats = clones.groupby('pitch_type')['Quality_Plus'].agg(['mean', 'std']).fillna(0)
        
        # Calculate pseudo-Sharpe ratio: (Mean / (StdDev + 1))
        stats['stability_score'] = stats['mean'] / (stats['std'] + 1)
        
        # Identify the most stable, high-ceiling pitch type for this specific chassis
        best_pitch_type = stats['stability_score'].idxmax()
        
        # 4. The "No Frankenstein" Logic
        optimal_cluster = clones[clones['pitch_type'] == best_pitch_type]
        best_clone = optimal_cluster.sort_values(by='Quality_Plus', ascending=False).iloc[0]

        top_pitch = str(best_clone['pitch_type'])
        top_score = round(float(best_clone['Quality_Plus']), 1)
        opt_axis = round(float(best_clone['spin_axis']), 1)
        opt_break_x = round(float(best_clone['pfx_x']), 1)
        opt_break_z = round(float(best_clone['pfx_z']), 1)
        clone_mlbid = int(best_clone['MLBID'])

        return {
            "status": "success",
            "recommended_pitch": top_pitch,
            "expected_quality_plus": top_score,
            "optimal_spin_axis": opt_axis,
            "optimal_pfx_x": opt_break_x,
            "optimal_pfx_z": opt_break_z,
            "apex_clone_mlbid": clone_mlbid,
            "reason": f"Filtered for arm slot ({t_z}z, {t_x}x) and velo delta. {top_pitch} offers the highest stable floor. Match clone #{clone_mlbid}'s specs for a {top_score} Q+."
        }
        
    except Exception as e:
        print(f"Recommender Error: {e}")
        return {
            "status": "error", 
            "recommended_pitch": "Engine Fault", 
            "reason": str(e)
        }

# ==========================================
# Test Execution Block (For local debugging)
# ==========================================
if __name__ == "__main__":
    # Simulate an incoming JSON payload mapped to a DataFrame
    sample_payload = pd.DataFrame([{
        "vaa": -5.2,
        "haa": 1.8,
        "release_extension": 6.8,
        "release_pos_z": 5.8,
        "release_pos_x": -2.1,
        "fastball_speed": 94.5,
        "release_speed": 85.2,
        "spin_axis": 210,
        "p_throws": "R"
    }])
    
    print("Testing Atlas OS Recommender Pipeline...")
    result = recommend_arsenal(sample_payload)
    print("\n[ API JSON RESPONSE ]")
    import json
    print(json.dumps(result, indent=2))
