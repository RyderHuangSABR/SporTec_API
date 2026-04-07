import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

# ==========================================
# 1. THE ATLAS CONFIGURATION (CONSTANTS)
# ==========================================

FEATURES = [
    'pitch_type', 'release_speed', 'effective_speed', 'release_spin_rate', 'spin_axis',
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
            print(f"  ✓ Loaded: {file.name} | Rows: {len(daily_df)}")
        except Exception as e:
            print(f"  ❌ Error loading {file.name}: {e}")
            
    master_df = pd.concat(df_list, ignore_index=True)
    if 'game_date' in master_df.columns:
        master_df = master_df.sort_values(by='game_date').reset_index(drop=True)
    
    print(f"\n✅ Atlas 2.0 Data Ingestion Complete. Total Pitches: {len(master_df)}")
    return master_df

# ==========================================
# 3. THE BIOMECHANICAL PRE-PROCESSOR
# ==========================================

def preprocess_atlas_data(df):
    """Cleans data and engineers the proprietary biological metrics."""
    # Group the pitches to avoid Statcast classification noise
    df['pitch_group'] = df['pitch_type'].map(PITCH_GROUPS)
    
    # Drop rows where critical trajectory data is missing
    df = df.dropna(subset=['pfx_x', 'pfx_z', 'plate_x', 'plate_z'])
    
    # --- PROPRIETARY ENGINEERING ---
    # Calculate movement_ratio (Example: hypotenuse of total break vs velocity)
    # Note: Adjust this to your specific secret formula
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['movement_ratio'] = df['total_break'] / df['release_speed']
    
    # Calculate reaction_time (Distance - Extension / Speed)
    # 55 feet is roughly the release point distance
    df['reaction_time'] = (55 - df['release_extension']) / df['effective_speed']
    
    return df

# ==========================================
# 4. THE XGBOOST CORE (TRAINING)
# ==========================================

def train_atlas_engine(df, target_col='contact_damage'):
    """Trains the XGBoost model to predict Contact Damage RMSE."""
    print("🧠 Initializing XGBoost Engine...")
    
    X = df[FEATURES]
    y = df[target_col]
    
    # One-hot encode categorical features (like pitch_type)
    X = pd.get_dummies(X, columns=['pitch_type'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # The actual algorithm
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Validation
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"🎯 Model Trained Successfully. Contact Damage RMSE: {rmse:.3f}")
    
    return model, X.columns

# ==========================================
# 5. THE DUGUOT API (FLASK ENDPOINT)
# ==========================================

app = Flask(__name__)
# In a real deployment, the model and features would be loaded globally here.
# global_model = None
# global_features = None

@app.route('/predict_pitch', methods=['POST'])
def predict_pitch():
    """
    The exact API endpoint Patrick Bailey hits from the iPad.
    Expects a JSON payload with the pitcher's chassis data.
    """
    try:
        incoming_data = request.get_json()
        
        # Convert incoming JSON to DataFrame
        pitch_df = pd.DataFrame([incoming_data])
        
        # Ensure all columns match the trained model (dummy columns included)
        # pitch_df = pitch_df.reindex(columns=global_features, fill_value=0)
        
        # Run the prediction
        # prediction = global_model.predict(pitch_df)[0]
        prediction = 0.321 # Hardcoded for demonstration
        
        # The payload that goes back to the dugout
        response = {
            "status": "success",
            "pitcher": incoming_data.get("pitcher_name", "Unknown"),
            "predicted_contact_damage_rmse": float(prediction),
            "recommendation": "DO NOT THROW" if prediction > 0.400 else "EXECUTE SEQUENCE"
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==========================================
# EXECUTION BLOCK (LOCAL TESTING)
# ==========================================
if __name__ == "__main__":
    print("🚀 Booting Atlas 2.0 System...")
    
    # 1. Load Data
    # raw_data = load_atlas_daily_pulls()
    
    # 2. Process
    # clean_data = preprocess_atlas_data(raw_data)
    
    # 3. Train
    # global_model, global_features = train_atlas_engine(clean_data)
    
    # 4. Launch API
    print("📡 Launching Dugout API on Port 5000...")
    app.run(host='0.0.0.0', port=5000)
