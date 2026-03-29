import os
import pandas as pd
import numpy as np
import duckdb
from huggingface_hub import hf_hub_download

# This ensures we only load what we need, when we need it
_DB_CONNECTION = None

def get_db_connection():
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
        DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
        
        # Download the file to a local temp path
        file_path = hf_hub_download(
            repo_id=DATA_REPO, 
            filename="reports/SABR_10_Year_Backtest_Leaderboard.csv", 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        
        # Initialize DuckDB in-memory but pointing to the file
        _DB_CONNECTION = duckdb.connect(database=':memory:')
        # Register the CSV as a virtual table to save RAM
        _DB_CONNECTION.execute(f"CREATE VIEW mlb_history AS SELECT * FROM read_csv_auto('{file_path}')")
    
    return _DB_CONNECTION

def recommend_arsenal(target_df):
    try:
        con = get_db_connection()
        
        # Extract target features from the iPad input
        t_vaa = target_df['vaa'].iloc[0]
        t_ext = target_df['release_extension'].iloc[0]
        t_speed = target_df['release_speed'].iloc[0]

        # STEP 1: Instead of KNN on the whole dataset, 
        # use DuckDB to filter for "Physical Clones" in SQL (Zero RAM usage)
        query = f"""
            SELECT MLBID, pitch_type, Quality_Plus 
            FROM mlb_history 
            WHERE vaa BETWEEN {t_vaa - 0.5} AND {t_vaa + 0.5}
            AND release_extension BETWEEN {t_ext - 0.2} AND {t_ext + 0.2}
            AND release_speed BETWEEN {t_speed - 2} AND {t_speed + 2}
            LIMIT 5000
        """
        clones = con.execute(query).df()

        if clones.empty:
            return {"recommended_pitch": "FB Variation", "reason": "Unique profile detected."}

        # STEP 2: Find what secondaries those physical clones throw successfully
        # Exclude fastballs (FF, SI, FC)
        secondaries = clones[~clones['pitch_type'].isin(['FF', 'SI', 'FC'])]
        
        if secondaries.empty:
            return {"recommended_pitch": "SL", "reason": "Standard high-VAA optimization."}

        # Group by pitch type and find the highest Quality+
        rec = secondaries.groupby('pitch_type')['Quality_Plus'].mean().sort_values(ascending=False)
        
        top_pitch = rec.index[0]
        top_score = round(float(rec.iloc[0]), 1)

        return {
            "recommended_pitch": top_pitch,
            "expected_quality_plus": top_score,
            "reason": f"Pitchers with your VAA and Extension generate elite {top_score} Q+ with this grip."
        }
    except Exception as e:
        print(f"Recommender Error: {e}")
        return {"recommended_pitch": "CH", "reason": "Fallback due to compute limits."}
