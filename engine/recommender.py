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
    
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    try:
        con = get_db_connection()
        
        # Extract pure physical data from the incoming iPad JSON
        t_vaa = float(target_df['vaa'].iloc[0])
        t_haa = float(target_df['haa'].iloc[0])
        t_ext = float(target_df['release_extension'].iloc[0])
        t_axis = float(target_df['spin_axis'].iloc[0])
        t_speed = float(target_df['release_speed'].iloc[0])

        # THE HOLY GRAIL: Weighted Euclidean Kinematic Distance
        # We weigh VAA and HAA massively (x10) because angle dictates the tunnel.
        # Extension is heavily weighted (x5) because it changes perceived reaction time.
        # Spin Axis difference is scaled down (x0.1) to normalize the 0-360 degree scale.
        
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
                    POWER((spin_axis - {t_axis}) * 0.1, 2)
                ) as kinematic_distance
            FROM mlb_history
            WHERE pitch_type NOT IN ('FF', 'SI', 'FC') -- Focus on secondaries / kill-shots
            ORDER BY kinematic_distance ASC
            LIMIT 100 -- We only extract the 100 closest physical clones in MLB history
        """
        
        clones = con.execute(query).df()

        if clones.empty:
            return {"recommended_pitch": "Unique Profile", "reason": "No historical clones match these kinematics."}

        # Instead of just picking a pitch type, we calculate the EXACT physical traits
        # required to throw the most optimized pitch from this arm slot.
        rec = clones.groupby('pitch_type').agg(
            avg_quality_plus=('Quality_Plus', 'mean'),
            opt_spin_axis=('spin_axis', 'mean'),
            opt_pfx_x=('pfx_x', 'mean'),
            opt_pfx_z=('pfx_z', 'mean')
        ).sort_values(by='avg_quality_plus', ascending=False)
        
        top_pitch = rec.index[0]
        top_score = round(float(rec.iloc[0]['avg_quality_plus']), 1)
        opt_axis = round(float(rec.iloc[0]['opt_spin_axis']), 1)
        opt_break_x = round(float(rec.iloc[0]['opt_pfx_x']), 1)
        opt_break_z = round(float(rec.iloc[0]['opt_pfx_z']), 1)

        return {
            "recommended_pitch": top_pitch,
            "expected_quality_plus": top_score,
            "optimal_spin_axis": opt_axis,
            "optimal_pfx_x": opt_break_x,
            "optimal_pfx_z": opt_break_z,
            "reason": f"Based on 100 physical clones (VAA, HAA, Ext), manipulating spin axis to {opt_axis}° generates an elite {top_score} Q+."
        }
    except Exception as e:
        print(f"Recommender Error: {e}")
        return {"recommended_pitch": "Error", "reason": "Math fault in distance calculation."}
