import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify

# ==========================================
# 1. THE ATLAS CONFIGURATION (CONSTANTS)
# ==========================================

FEATURES = [
    'release_speed', 'effective_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z', 'release_extension', 
    'plate_x', 'plate_z', 'vaa', 'haa', 'commit_x', 'commit_z', 'reaction_time', 'movement_ratio'
]

PITCH_GROUPS = {
    "FF": "Fastball", "SI": "Sinker", "FC": "Cutter", "CH": "Changeup",
    "FS": "SplitterFork", "FO": "SplitterFork", "ST": "SweeperSlider",
    "SL": "SweeperSlider", "SW": "SweeperSlider", "CU": "Curveball",
    "KC": "Curveball", "CS": "Curveball"
}

# ==========================================
# 2. THE DATA INGESTION ENGINE (PARQUET)
# ==========================================

def load_atlas_daily_pulls(base_dir="Atlas_Pitching_Data/daily_pulls"):
    """Scrapes the directory for daily Parquet files and concatenates them."""
    data_path = Path(base_dir)
    parquet_files = list(data_path.glob("Pitches_*.parquet"))
    
    if not parquet_files:
        print(f"⚠️ Warning: No daily pulls found in {base_dir}")
        return pd.DataFrame()
        
    print(f"📊 Discovered {len(parquet_files)} daily parquet files. Initiating ingestion...")
    
    df_list = []
    for file in parquet_files:
        try:
            file_date = file.stem.split('_')[1]
            daily_df = pd.read_parquet(file)
            if 'game_date' not in daily_df.columns:
                daily_df['game_date'] = pd.to_datetime(file_date)
            df_list.append(daily_df)
        except Exception as e:
            print(f"  ❌ Error loading {file.name}: {e}")
            
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Atlas 2.0 Data Ingestion Complete. Total Pitches: {len(master_df)}")
    return master_df

# ==========================================
# 3. THE BIOMECHANICAL PRE-PROCESSOR
# ==========================================

def preprocess_atlas_data(df):
    """Cleans data and engineers the proprietary biological metrics."""
    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS)
    df = df.dropna(subset=['pfx_x', 'pfx_z', 'plate_x', 'plate_z'])
    
    # Proprietary Engineering
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_ratio'] = df['total_break'] / df['release_speed']
    df['reaction_time'] = (55 - df['release_extension']) / df['effective_speed']
    
    return df

# ==========================================
# 4. XGBOOST CORE (HEAVY ARTILLERY)
# ==========================================

def train_xgboost_engine(df, target_col='contact_damage'):
    """Trains the XGBoost model to predict Contact Damage RMSE."""
    print("🧠 Initializing XGBoost Engine...")
    
    X = df[FEATURES].fillna(0)
    y = df[target_col].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"🎯 XGBoost Trained Successfully. Damage RMSE: {rmse:.3f}")
    
    return model

# ==========================================
# 5. WEIGHTED 1-NN CORE (THE SCALPEL)
# ==========================================

def build_weighted_recommender(xgb_model, df):
    """Extracts XGBoost weights to bypass the Curse of Dimensionality for 1-NN."""
    print("🧬 Initializing Weighted 1-NN Clone Engine...")
    
    # 1. Extract and normalize XGBoost feature importances
    xgb_weights = xgb_model.feature_importances_
    normalized_weights = xgb_weights / np.sum(xgb_weights)
    
    # 2. Apply weights to the raw dataset
    X_raw = df[FEATURES].fillna(0).copy()
    X_weighted = X_raw * normalized_weights
    
    # 3. Train the Nearest Neighbors model on the stretched/shrunk data
    nn_model = NearestNeighbors(n_neighbors=2, metric='euclidean') # n=2 because index 0 is the pitch itself
    nn_model.fit(X_weighted)
    
    print("✅ Clone Engine Online.")
    return nn_model, normalized_weights

def find_pitch_clone(target_features, nn_model, weights, original_df):
    """Finds the closest historical pitch using the weighted engine."""
    target_weighted = np.array(target_features) * weights
    distance, index = nn_model.kneighbors([target_weighted])
    
    # index[0][1] gets the *first* nearest neighbor that isn't the exact same row
    clone_idx = index[0][1]
    clone_pitch = original_df.iloc[clone_idx]
    
    return clone_pitch, distance[0][1]

# ==========================================
# 6. THE DUGOUT API (FLASK ENDPOINT)
# ==========================================

app = Flask(__name__)

# Globals for the API to access the trained models
atlas_df = None
xgb_engine = None
nn_engine = None
feature_weights = None

@app.route('/predict_pitch', methods=['POST'])
def predict_pitch():
    """The dual-engine payload for Patrick Bailey's iPad."""
    try:
        incoming_data = request.get_json()
        pitch_df = pd.DataFrame([incoming_data]).reindex(columns=FEATURES, fill_value=0)
        
        # 1. Get XGBoost Damage Prediction
        predicted_damage = xgb_engine.predict(pitch_df)[0]
        
        # 2. Get Weighted 1-NN Historical Clone
        target_array = pitch_df.iloc[0].values
        clone, distance = find_pitch_clone(target_array, nn_engine, feature_weights, atlas_df)
        
        # 3. Assemble the Ultimate Payload
        response = {
            "status": "success",
            "pitcher": incoming_data.get("pitcher_name", "Unknown"),
            "xgboost_analysis": {
                "predicted_contact_damage_rmse": float(predicted_damage),
                "recommendation": "DO NOT THROW" if predicted_damage > 0.400 else "EXECUTE SEQUENCE"
            },
            "historical_clone_analysis": {
                "clone_pitcher": clone.get("pitcher_name", "Unknown Historical"),
                "clone_pitch_type": clone.get("pitch_type", "Unknown"),
                "similarity_distance": float(distance),
                "historical_result": clone.get("events", "Unknown Outcome")
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("🚀 Booting Atlas 2.0 Master System...")
    # NOTE: Uncomment to run live
    
    # 1. Load & Process Data
    # atlas_df = preprocess_atlas_data(load_atlas_daily_pulls())
    
    # 2. Train Both Engines
    # xgb_engine = train_xgboost_engine(atlas_df)
    # nn_engine, feature_weights = build_weighted_recommender(xgb_engine, atlas_df)
    
    # 3. Launch the API
    # app.run(host='0.0.0.0', port=5000)
