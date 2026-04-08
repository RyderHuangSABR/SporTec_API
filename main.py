# main.py
import os
import secrets
import logging
import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.recommender import recommend_arsenal

# --- CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas_api")

# Initialize Local Analytical Database
db = duckdb.connect("atlas_application.db")

db.execute("""
    CREATE TABLE IF NOT EXISTS api_clients (
        api_key TEXT PRIMARY KEY,
        client_name TEXT,
        tier TEXT
    );
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

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Atlas Pitching Analytics API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key")

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

# --- DEPENDENCIES & SERVICES ---
def authenticate_client(api_key: str = Security(api_key_header)):
    """Validates the incoming API key against the database."""
    result = db.execute("SELECT client_name FROM api_clients WHERE api_key = ?", [api_key]).fetchone()
    if not result:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")
    return result[0]

def log_application_telemetry(client_name: str, pitch_data: dict, recommendation: dict):
    """Asynchronously logs API usage and prediction data to DuckDB."""
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
        logger.error(f"Telemetry logging failed: {e}")

# --- ENDPOINTS ---
@app.post("/api/v1/predict")
@limiter.limit("10/minute") 
async def predict_pitch(
    request: Request, 
    pitch: TargetPitch, 
    background_tasks: BackgroundTasks, 
    client_name: str = Security(authenticate_client)
):
    """Main prediction endpoint for generating pitch recommendations."""
    logger.info(f"Prediction requested by client: {client_name}")
    
    target_df = pd.DataFrame([pitch.model_dump()])
    result = recommend_arsenal(target_df)
    
    background_tasks.add_task(log_application_telemetry, client_name, pitch.model_dump(), result)
    
    return {
        "status": "success",
        "client_id": client_name,
        "data": result
    }

@app.post("/admin/generate_key")
async def generate_api_key(req: APIKeyRequest):
    """Administrative endpoint to provision new API clients."""
    expected_password = os.getenv("ATLAS_ADMIN_SECRET")
    if not expected_password or req.admin_password != expected_password:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid credentials.")
    
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
        raise HTTPException(status_code=500, detail="Database fault during key generation.")

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    """System health monitor."""
    return {"status": "healthy", "service": "Atlas Application API"}
