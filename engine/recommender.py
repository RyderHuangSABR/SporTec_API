import os
import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download

_DB_CONNECTION = None

def get_db_connection():
    """Bootstraps the DuckDB in-memory vault on first load."""
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
        DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
        
        print("⚙️ Fetching 10-Year Backtest from HuggingFace Vault...")
        # Pull the historical Parquet/CSV from the vault
        file_path = hf_hub_download(
            repo_id=DATA_REPO, 
            filename="reports/SABR_10_Year_Backtest_Leaderboard.csv", 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        
        # Initialize DuckDB in-memory for zero-latency queries
        _DB_CONNECTION = duckdb.connect(database=':memory:')
        _DB_CONNECTION.execute(f"CREATE VIEW mlb_history AS SELECT * FROM read_csv_auto('{file_path}')")
        print("💎 DuckDB Engine Armed and Ready.")
    
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    """The Atlas OS Core: Weighted KNN via DuckDB SQL."""
    try:
        con = get_db_connection()
        
        # 1. Extract pure physical data from the incoming JSON payload
        t_vaa = float(target_df['vaa'].iloc[0])
        t_haa = float(target_df['haa'].iloc[0])
        t_ext = float(target_df['release_extension'].iloc[0])
        t_axis = float(target_df['spin_axis'].iloc[0])
        t_speed = float(target_df['release_speed'].iloc[0])
        
        # Critical: We must extract handedness so we don't mirror-match lefties to righties
        t_hand = str(target_df['p_throws'].iloc[0]).upper()

        # 2. THE HOLY GRAIL: Weighted Euclidean Kinematic Distance
        # - VAA/HAA (x10) because angle dictates the tunnel.
        # - Extension (x5) because it changes perceived reaction time.
        # - Circular Spin Axis Math: calculates the shortest distance around a 360-degree wheel.
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
                    POWER((vaa - {t_vaa}) * 10.0, 2) + 
                    POWER((haa - {t_haa}) * 10.0, 2) + 
                    POWER((release_extension - {t_ext}) * 5.0, 2) + 
                    POWER((release_speed - {t_speed}) * 1.0, 2) +
                    POWER(LEAST(ABS(spin_axis - {t_axis}), 360 - ABS(spin_axis - {t_axis})) * 0.1, 2)
                ) as kinematic_distance
            FROM mlb_history
            WHERE pitch_type NOT IN ('FF', 'SI', 'FC') -- Focus on secondaries / kill-shots
            AND p_throws = '{t_hand}' -- Enforce handedness quarantine
            ORDER BY kinematic_distance ASC
            LIMIT 100 -- Extract the 100 closest physical clones in MLB history
        """
        
        clones = con.execute(query).df()

        if clones.empty:
            return {
                "recommended_pitch": "Unique Profile", 
                "reason": f"No historical {t_hand}HP clones match these kinematics. Profile is structurally unprecedented."
            }

        # 3. The "No Frankenstein" Logic
        # First, find the pitch type that performs best *on average* in this exact kinematic slot.
        best_pitch_type = clones.groupby('pitch_type')['Quality_Plus'].mean().idxmax()
        
        # Next, isolate ONLY that pitch type from our nearest neighbors.
        optimal_cluster = clones[clones['pitch_type'] == best_pitch_type]
        
        # Finally, grab the literal #1 best historical pitch from that cluster.
        # We don't average the breaks. We steal the exact blueprint of the apex survivor.
        best_clone = optimal_cluster.sort_values(by='Quality_Plus', ascending=False).iloc[0]

        top_pitch = str(best_clone['pitch_type'])
        top_score = round(float(best_clone['Quality_Plus']), 1)
        opt_axis = round(float(best_clone['spin_axis']), 1)
        opt_break_x = round(float(best_clone['pfx_x']), 1)
        opt_break_z = round(float(best_clone['pfx_z']), 1)
        clone_mlbid = int(best_clone['MLBID'])

        return {
            "recommended_pitch": top_pitch,
            "expected_quality_plus": top_score,
            "optimal_spin_axis": opt_axis,
            "optimal_pfx_x": opt_break_x,
            "optimal_pfx_z": opt_break_z,
            "apex_clone_mlbid": clone_mlbid,
            "reason": f"Analyzed 100 {t_hand}HP clones. The {top_pitch} dominates this kinematic slot. Matching clone #{clone_mlbid}'s exact metrics (Axis: {opt_axis}°, Break: {opt_break_x}x, {opt_break_z}z) yields an elite {top_score} Q+."
        }
        
    except Exception as e:
        print(f"Recommender Error: {e}")
        return {"recommended_pitch": "Error", "reason": "Engine fault. Verify JSON payload includes p_throws."}
