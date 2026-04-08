import os
import secrets
import duckdb
from fastapi import FastAPI, HTTPException, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.recommender import recommend_arsenal

# --- DUCKDB SETUP ---
# Connect to a local file database
db = duckdb.connect("atlas_vault.db")

# Initialize tables if they don't exist
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

def get_api_key_from_request(request: Request):
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(key_func=get_api_key_from_request)

app = FastAPI(title="Atlas Pitching Engine", version="4.0-DuckDB-Broker")
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

def get_client_identity(api_key: str = Security(api_key_header)):
    # Query DuckDB for the key
    result = db.execute("SELECT client_name FROM api_clients WHERE api_key = ?", [api_key]).fetchone()
    
    if not result:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")
    
    return result[0]

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

class MintRequest(BaseModel):
    client_name: str
    tier: str
    admin_password: str

# --- THE TELEMETRY HARVESTER ---
def log_hexcore_telemetry(client_name: str, pitch_data: dict, recommendation: dict):
    """Silently harvests telemetry into DuckDB."""
    try:
        db.execute("""
            INSERT INTO telemetry_logs 
            (client_name, input_vaa, input_speed, input_p_throws, recommended_pitch, kinematic_distance, apex_clone_mlbid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            client_name,
            pitch_data["vaa"],
            pitch_data["release_speed"],
            pitch_data["p_throws"],
            recommendation.get("recommended_pitch", "Error"),
            recommendation.get("kinematic_distance", 999.9),
            recommendation.get("apex_clone_mlbid", 0)
        ])
    except Exception as e:
        print(f"Silent Telemetry Error: {e}")

@app.post("/predict")
@limiter.limit("10/minute") 
async def predict(
    request: Request, 
    pitch: TargetPitch, 
    background_tasks: BackgroundTasks, 
    client_name: str = Security(get_client_identity)
):
    print(f"⚙️ {client_name} requested a 1-NN DuckDB scan. Limit: OK.")
    
    target_df = pd.DataFrame([pitch.model_dump()])
    result = recommend_arsenal(target_df)
    
    background_tasks.add_task(log_hexcore_telemetry, client_name, pitch.model_dump(), result)
    
    return {
        "status": "success",
        "client_billed": client_name,
        "prediction": result
    }

@app.post("/admin/mint_key")
async def mint_api_key(req: MintRequest):
    expected_password = os.getenv("ATLAS_MASTER_KEY")
    if not expected_password or req.admin_password != expected_password:
        raise HTTPException(status_code=403, detail="Nice try. Access Denied.")
    
    new_api_key = f"atl_{secrets.token_hex(16)}"
    
    try:
        db.execute(
            "INSERT INTO api_clients (api_key, client_name, tier) VALUES (?, ?, ?)",
            [new_api_key, req.client_name, req.tier]
        )
        
        return {
            "status": "success",
            "message": "Key injected into DuckDB vault.",
            "client_name": req.client_name,
            "api_key": new_api_key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database fault: {e}")

@app.get("/")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"message": "Atlas 1-NN Engine is Live (Powered by DuckDB)."}
