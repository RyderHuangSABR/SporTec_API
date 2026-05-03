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
from sklearn.preprocessing import StandardScaler

from engine.recommender import recommend_arsenal
from engine.loader import load_atlas_data

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas_api")

# --- LIFESPAN (Runs once when server starts) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up: Downloading Hugging Face data...")
    try:
        # 1. Download the data using your loader
        df_master, df_dict = load_atlas_data()
        
        # 2. Store it in app.state so the whole app can access it in RAM
        app.state.df = df_master
        app.state.target_dict = df_dict
        
        # 3. Create your Scaler and Weights
        app.state.scaler = StandardScaler() 
        # (If your scaler needs fitting, you'd do it here like: app.state.scaler.fit(df_master[['col1', 'col2']]))
        app.state.weights = [1.0, 1.0, 1.0, 1.0, 1.0] # Replace with your actual weights
        
        logger.info("✅ ML Brain loaded into RAM! Opening for traffic...")
    except Exception as e:
        logger.error(f"❌ Failed to load ML assets: {e}")
        
    yield
    
    logger.info("🛑 Shutting down server...")

# --- APP INIT ---
app = FastAPI(
    title="Atlas Pitching Analytics API",
    version="2.0.0",
    lifespan=lifespan  # 🚨 Added the lifespan here!
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

# Pull your Master Key from Render's Environment Vault
MASTER_KEY = os.getenv("API_KEY", "6YHN4RFV3edc@")

def authenticate_client(api_key: str = Security(api_key_header)):
    # Master Key Override (Survives Render Restarts!)
    if api_key == MASTER_KEY:
        return "Atlas Admin"

    # Client Key Check (Looks in DuckDB for generated keys)
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
            recommendation.get("kinematic_distance", 999.9),
            recommendation.get("apex_clone_mlbid", 0)
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
        "service": "Atlas API"
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
        df_input = pd.DataFrame([pitch.model_dump()])
        
        # 🚨 FIX: Pass ALL 5 required parameters from the app state
        result = recommend_arsenal(
            target_features=df_input,
            target_dict=request.app.state.target_dict,
            scaler=request.app.state.scaler,
            weights=request.app.state.weights,
            df=request.app.state.df
        )

        background_tasks.add_task(
            log_application_telemetry,
            client_name,
            pitch.model_dump(),
            result
        )

        return {
            "status": "success",
            "client_id": client_name,
            "data": result
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/admin/generate_key")
async def generate_api_key(req: APIKeyRequest):
    expected_password = os.getenv("API_KEY")

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
