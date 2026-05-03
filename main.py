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

from engine.recommender import recommend_arsenal

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas_api")

# --- LIFESPAN: SIMPLIFIED ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # We no longer need to download the Parquet file! 
    # Just ensure the GMM CSV is in the /atlas folder.
    logger.info("🚀 Atlas API Starting Up...")
    if not os.path.exists("atlas/gmm_pitch_profiles.csv"):
        logger.error("❌ CRITICAL: gmm_pitch_profiles.csv not found in /atlas directory!")
    else:
        logger.info("✅ GMM Profiles found. Ready for deployment.")
    yield

# --- APP INIT ---
app = FastAPI(
    title="Atlas Pitching Analytics API",
    version="2.1.0",
    lifespan=lifespan
)

# --- RATE LIMITING & CORS (Keep these as they were) ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE INIT (Telemetry stays local) ---
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
            input_speed DOUBLE,
            recommended_pitch_type TEXT,
            kinematic_distance DOUBLE,
            matched_mlbid INTEGER
        );
    """)
    return db

db = init_db()

# --- SECURITY ---
api_key_header = APIKeyHeader(name="X-API-Key")
MASTER_KEY = os.getenv("API_KEY", "6YHN4RFV3edc@")

def authenticate_client(api_key: str = Security(api_key_header)):
    if api_key == MASTER_KEY:
        return "Atlas Admin"
    result = db.execute("SELECT client_name FROM api_clients WHERE api_key = ?", [api_key]).fetchone()
    if not result:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return result[0]

# --- MODELS ---
class TargetPitch(BaseModel):
    # The fields your frontend sends
    release_speed: float
    pfx_x: float
    pfx_z: float
    release_pos_x: float
    release_pos_z: float
    release_extension: float

# --- TELEMETRY ---
def log_application_telemetry(client_name: str, pitch_data: dict, result: dict):
    try:
        # Adjusted to match the new GMM return structure
        identity = result.get("identity", {})
        db.execute("""
            INSERT INTO telemetry_logs 
            (client_name, input_speed, recommended_pitch_type, kinematic_distance, matched_mlbid)
            VALUES (?, ?, ?, ?, ?)
        """, [
            client_name,
            pitch_data.get("release_speed", 0.0),
            identity.get("matched_pitch_type", "Unknown"),
            result.get("distance", 0.0),
            identity.get("matched_pitcher_id", 0)
        ])
    except Exception as e:
        logger.error(f"Telemetry failed: {e}")

# --- ROUTES ---

@app.get("/")
async def root():
    return {"status": "ok", "engine": "GMM-Centroid-v2"}

@app.post("/api/v1/predict")
@limiter.limit("20/minute") # We can increase limit because it's faster now!
async def predict_pitch(
    request: Request,
    pitch: TargetPitch,
    background_tasks: BackgroundTasks,
    client_name: str = Security(authenticate_client)
):
    try:
        df_input = pd.DataFrame([pitch.model_dump()])
        
        # This calls your lightning-fast GMM engine
        result = recommend_arsenal(df_input)

        background_tasks.add_task(
            log_application_telemetry,
            client_name,
            pitch.model_dump(),
            result
        )

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# (Keep admin routes as they were...)
