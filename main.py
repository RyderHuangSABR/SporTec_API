# main.py
import os
import secrets
import logging
import duckdb
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sklearn.preprocessing import StandardScaler

# --- IMPORT YOUR NEW ENGINE LOGIC ---
from engine.loader import load_atlas_data, get_models_for_pitch
from engine.recommender import recommend_arsenal, preprocess_atlas_data
from engine.features import FEATURES  # Needed to align the XGBoost weights

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas_api")

# --- LIFESPAN (THE CLOUD ML BRIDGE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Downloads Master Data from Hugging Face and initializes the Scaler globally.
    Models are dynamically cached during runtime via loader.py.
    """
    logger.info("Booting up Atlas OS Cloud ML Core...")
    
    try:
        # 1. Fetch Parquets from Hugging Face via loader.py
        df_master, df_dict = load_atlas_data()
        
        # 2. Preprocess data to engineer kinematic features
        logger.info("Preprocessing Master Biomechanical Dataset...")
        clean_df = preprocess_atlas_data(df_master)
        
        # 3. Fit the Universal Scaler
        logger.info("Fitting standard scaler for 1-NN Euclidean calculations...")
        scaler = StandardScaler()
        scaler.fit(clean_df[FEATURES].fillna(0))
        
        # 4. Save to global state
        app.state.clean_df = clean_df
        app.state.df_dict = df_dict
        app.state.scaler = scaler
        
        logger.info("✅ Atlas OS Cloud Brain loaded into RAM.")
        yield
        
    except Exception as e:
        logger.error(f"Failed to load ML Brain from Hugging Face: {e}")
        raise e
        
    finally:
        logger.info("Shutting down Atlas OS...")

# --- APP INIT ---
app = FastAPI(
    title="Atlas Pitching Analytics API",
    version="3.0.0",
    lifespan=lifespan 
)

# --- RATE LIMITING ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE INIT ---
def init_db():
    db = duckdb.connect("atlas_application.db")
    db.execute("""
        CREATE TABLE IF NOT EXISTS api_clients (
            api_key TEXT PRIMARY KEY,
            client_name TEXT,
            tier TEXT
        );
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            client_name TEXT,
            input_vaa DOUBLE,
            input_speed DOUBLE,
            input_p_throws TEXT,
            recommended_pitch TEXT,
            kinematic_distance DOUBLE,
            apex_clone_mlbid INTEGER
        );
    """)
    return db

db = init_db()

# --- SECURITY ---
api_key_header = APIKeyHeader(name="X-API-Key")

def authenticate_client(api_key: str = Security(api_key_header)):
    result = db.execute(
        "SELECT client_name FROM api_clients WHERE api_key = ?",
        [api_key]
    ).fetchone()

    if not result:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return result[0]

# --- MODELS ---
class TargetPitch(BaseModel):
    p_throws: str
    vaa: float
    haa: float
    release_extension: float
    release_pos_z: float
    release_pos_x: float
    fastball_speed: float
    release_speed: float
    spin_axis: float
    pfx_x: float 
    pfx_z: float
    plate_x: float
    plate_z: float
    effective_speed: float
    pitch_type: str

class APIKeyRequest(BaseModel):
    client_name: str
    tier: str
    admin_password: str

# --- TELEMETRY ---
def log_application_telemetry(client_name: str, pitch_data: dict, recommendation: dict):
    try:
        db.execute("""
            INSERT INTO telemetry_logs 
            (client_name, input_vaa, input_speed, input_p_throws, recommended_pitch, kinematic_distance, apex_clone_mlbid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            client_name,
            pitch_data.get("vaa", 0.0),
            pitch_data.get("release_speed", 0.0),
            pitch_data.get("p_throws", "U"),
            recommendation.get("recommended_pitch", "Error"),
            recommendation.get("distance", 999.9), 
            recommendation.get("clone_pitch", {}).get("pitcher", 0) if recommendation.get("clone_pitch") is not None else 0
        ])
    except Exception as e:
        logger.error(f"Telemetry failed: {e}")

# --- ROUTES ---

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Atlas API",
        "data_source": "Hugging Face"
    }

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "service": "Atlas API",
        "hf_data_loaded": hasattr(request.app.state, 'clean_df')
    }

@app.post("/api/v1/predict")
@limiter.limit("10/minute")
async def predict_pitch(
    request: Request,
    pitch: TargetPitch,
    background_tasks: BackgroundTasks,
    client_name: str = Security(authenticate_client)
):
    logger.info(f"Prediction request from: {client_name}")

    try:
        # 1. Retrieve Global ML State
        scaler = request.app.state.scaler
        historical_df = request.app.state.clean_df
        
        # 2. Format request and Preprocess
        target_dict = pitch.model_dump()
        target_df = pd.DataFrame([target_dict])
        target_df = preprocess_atlas_data(target_df)

        # 3. Pull Pre-Trained XGBoost Models from Hugging Face Cache
        target_pitch_type = target_dict.get("pitch_type", "FF")
        models = get_models_for_pitch(target_pitch_type)
        
        if not models:
            raise HTTPException(status_code=400, detail=f"No models deployed for pitch type: {target_pitch_type}")

        # 4. Extract Weights from Engine B (Contact Damage Booster)
        booster_b = models["B"]
        # .get_score() extracts feature importance from native boosters
        scores = booster_b.get_score(importance_type='gain') 
        
        # Map scores to the feature array safely (defaulting to tiny non-zero weight if missing)
        raw_weights = np.array([scores.get(feat, 1e-4) for feat in FEATURES])
        normalized_weights = raw_weights / np.sum(raw_weights)

        # 5. Run the Recommender Pipeline
        result = recommend_arsenal(
            target_df=target_df,
            target_dict=target_dict,
            scaler=scaler,
            weights=normalized_weights,
            df=historical_df
        )

        # 6. JSON Serialization
        safe_result = {
            "distance": float(result["distance"]) if result.get("distance") is not None else None,
            "error": result.get("error"),
            "clone_pitch": result.get("clone_pitch").to_dict() if result.get("clone_pitch") is not None else None,
            "arsenal": result.get("arsenal").to_dict(orient="records") if not result.get("arsenal").empty else None,
        }

        # 7. Telemetry Fire-and-Forget
        background_tasks.add_task(
            log_application_telemetry,
            client_name,
            target_dict,
            safe_result
        )

        return {
            "status": "success",
            "client_id": client_name,
            "data": safe_result
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/generate_key")
async def generate_api_key(req: APIKeyRequest):
    expected_password = os.getenv("ATLAS_ADMIN_SECRET")

    if not expected_password or req.admin_password != expected_password:
        raise HTTPException(status_code=403, detail="Forbidden")

    new_api_key = f"atl_{secrets.token_hex(16)}"

    try:
        db.execute(
            "INSERT INTO api_clients (api_key, client_name, tier) VALUES (?, ?, ?)",
            [new_api_key, req.client_name, req.tier]
        )

        return {
            "status": "success",
            "client_name": req.client_name,
            "api_key": new_api_key
        }

    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
