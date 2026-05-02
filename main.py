# main.py
import os
import secrets
import logging
import duckdb
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import your ML engine functions
from engine.recommender import (
    recommend_arsenal, 
    preprocess_atlas_data, 
    train_xgboost_model, 
    prepare_distance_metrics
)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas_api")

# --- LIFESPAN (THE ML BRIDGE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs exactly once when the server starts. 
    It loads your brain into RAM so it's lightning-fast for API clients.
    """
    logger.info("Booting up Atlas OS Machine Learning Core...")
    
    try:
        # 1. Load the historical Statcast dataset (Update path as needed)
        # Assuming you have a local CSV of statcast data for the 1-NN search
        logger.info("Loading Statcast dataset...")
        raw_df = pd.read_csv("data/statcast_historical.csv") 
        
        # 2. Preprocess the data to engineer your kinematic features
        logger.info("Preprocessing biomechanical data...")
        clean_df = preprocess_atlas_data(raw_df)
        
        # 3. Load or Train your XGBoost Model 
        # (If you download your HF model, load it here. For now, we train on boot)
        logger.info("Initializing XGBoost weights...")
        xgb_model = train_xgboost_model(clean_df, target_col='contact_damage')
        
        # 4. Extract Scaler and Normalized Weights
        scaler, weights = prepare_distance_metrics(xgb_model, clean_df)
        
        # 5. Save everything to the app's global state
        app.state.clean_df = clean_df
        app.state.scaler = scaler
        app.state.weights = weights
        app.state.xgb_model = xgb_model
        
        logger.info("✅ Atlas OS Brain loaded securely into RAM.")
        yield
        
    except Exception as e:
        logger.error(f"Failed to load ML Brain on startup: {e}")
        raise e
        
    finally:
        logger.info("Shutting down Atlas OS...")

# --- APP INIT ---
app = FastAPI(
    title="Atlas Pitching Analytics API",
    version="2.0.0",
    lifespan=lifespan # Connects the ML loader to the app
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
    # Add these based on your preprocess_atlas_data needs
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
            recommendation.get("distance", 999.9), # Fixed from kinematic_distance
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
        "docs": "/docs"
    }

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "service": "Atlas API",
        "ml_loaded": hasattr(request.app.state, 'xgb_model')
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
        # 1. Retrieve the ML State from the App
        scaler = request.app.state.scaler
        weights = request.app.state.weights
        historical_df = request.app.state.clean_df
        
        # 2. Format the incoming request
        target_dict = pitch.model_dump()
        target_df = pd.DataFrame([target_dict])
        
        # 3. Preprocess target to create movement_ratio, total_break, etc.
        target_df = preprocess_atlas_data(target_df)

        # 4. Pass ALL required arguments to the engine (This fixes the crash)
        result = recommend_arsenal(
            target_df=target_df,
            target_dict=target_dict,
            scaler=scaler,
            weights=weights,
            df=historical_df
        )

        # 5. Clean pandas objects to JSON serializable formats
        safe_result = {
            "distance": float(result["distance"]) if result.get("distance") is not None else None,
            "error": result.get("error"),
            "clone_pitch": result.get("clone_pitch").to_dict() if result.get("clone_pitch") is not None else None,
            "arsenal": result.get("arsenal").to_dict(orient="records") if not result.get("arsenal").empty else None,
        }

        # 6. Log telemetry in background
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
