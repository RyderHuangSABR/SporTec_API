import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download

_HISTORY_CACHE = None

def get_mlb_history():
    global _HISTORY_CACHE
    if _HISTORY_CACHE is not None: return _HISTORY_CACHE
    HF_TOKEN = os.getenv("HF_TOKEN")
    DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"
    file_path = hf_hub_download(
        repo_id=DATA_REPO, 
        filename="reports/SABR_10_Year_Backtest_Leaderboard.csv", 
        repo_type="dataset", token=HF_TOKEN
    )
    _HISTORY_CACHE = pd.read_csv(file_path)
    return _HISTORY_CACHE

def recommend_arsenal(target_fastball_df):
    mlb_history_df = get_mlb_history()
    features = ['vaa', 'haa', 'release_extension', 'spin_axis', 'release_speed', 'pfx_x', 'pfx_z']
    fastballs_only = mlb_history_df[mlb_history_df['pitch_type'].isin(['FF', 'SI', 'FC'])].dropna(subset=features)
    
    if fastballs_only.empty: return {"recommended_pitch": "None"}
    
    scaler = StandardScaler()
    X_matrix = scaler.fit_transform(fastballs_only[features])
    knn = NearestNeighbors(n_neighbors=50, metric='euclidean')
    knn.fit(X_matrix)
    
    target_vector = target_fastball_df[features].mean().values.reshape(1, -1)
    if np.isnan(target_vector).any(): return {"recommended_pitch": "Insufficient Data"}
        
    target_scaled = scaler.transform(target_vector)
    distances, indices = knn.kneighbors(target_scaled)
    clone_mlbids = fastballs_only.iloc[indices[0]]['MLBID'].unique()
    
    clone_arsenals = mlb_history_df[mlb_history_df['MLBID'].isin(clone_mlbids)]
    secondaries = clone_arsenals[~clone_arsenals['pitch_type'].isin(['FF', 'SI', 'FC'])]
    
    recommendations = secondaries.groupby('pitch_type').agg(
        Usage_Count=('pitch_type', 'count'),
        Avg_Quality_Plus=('Quality_Plus', 'mean')
    ).reset_index()
    recommendations = recommendations[recommendations['Usage_Count'] > 500].sort_values(by='Avg_Quality_Plus', ascending=False)
    
    if recommendations.empty: return {"recommended_pitch": "None", "reason": "No match."}
        
    return {
        "recommended_pitch": recommendations.iloc[0]['pitch_type'],
        "expected_quality_plus": round(recommendations.iloc[0]['Avg_Quality_Plus'], 1),
        "reason": f"Pitchers with identical Fastball VAA, Extension, and Spin generated a {round(recommendations.iloc[0]['Avg_Quality_Plus'], 1)} Quality+ with this pitch."
    }
