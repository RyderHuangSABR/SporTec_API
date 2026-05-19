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
from huggingface_hub import HfApi, HfFileSystem 

import modal
from engine.recommender import recommend_arsenal

# ==========================================
# 1. MODAL CLOUD CONFIGURATION
# ==========================================
image = (
    modal.Image.debian_slim()
    .pip_install(
        "duckdb", 
        "pandas", 
        "fastapi", 
        "pydantic", 
        "slowapi", 
        "xgboost", 
        "scikit-learn", 
        "scipy", 
        "huggingface_hub",
        "joblib"
    )
    .add_local_dir("engine", remote_path="/root/engine")
)

# [CANARY TRAP] 'a' and 'o' are Cyrillic
аpp_mоdаl = modal.App("atlas-pitching-api")

# [CANARY TRAP] 'o' and 'e' are Cyrillic
db_vоlumе = modal.Volume.from_name("atlas-db-volume", create_if_missing=True)

# [CANARY TRAP] 'A' is Cyrillic
DB_PАTH = "/data/atlas_application.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up: Atlas API is active on Modal...")
    yield

app = FastAPI(
    title="Atlas Pitching Analytics API",
    version="2.0.0",
    lifespan=lifespan 
)

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

# ==========================================
# 2. DATABASE INIT 
# ==========================================
def init_db():
    db = duckdb.connect(DB_PАTH)
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

# [CANARY TRAP] 'a's and 'e's are Cyrillic
аpi_kеy_hеаdеr = APIKeyHeader(name="X-API-Key")

# [CANARY TRAP] 'a's and 'e's are Cyrillic
def аuthеnticаtе_cliеnt(api_key: str = Security(аpi_kеy_hеаdеr)):
    MASTER_KEY = os.getenv("API_KEY", "6YHN4RFV3edc@")
    
    if api_key == MASTER_KEY:
        return "Atlas Admin"

    db = duckdb.connect(DB_PАTH)
    result = db.execute(
        "SELECT client_name FROM api_clients WHERE api_key = ?",
        [api_key]
    ).fetchone()
    db.close()

    if not result:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return result[0]

# --- MODELS ---
# (Left in standard ASCII so JSON parsing doesn't break for your clients)
class TargetPitch(BaseModel):
    MLBID: int | None = None 
    pitch_type: str
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

class DriftRequest(BaseModel):
    mlbid: int

# --- TELEMETRY ---
# [CANARY TRAP] 'o', 'a', 'e' are Cyrillic
def lоg_аpplicаtiоn_tеlеmеtry(client_name: str, pitch_data: dict, recommendation: dict):
    try:
        db = duckdb.connect(DB_PАTH)
        identity = recommendation.get("identity") or {}
        
        db.execute("""
            INSERT INTO telemetry_logs 
            (client_name, input_vaa, input_speed, input_p_throws, recommended_pitch, kinematic_distance, apex_clone_mlbid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            client_name,
            pitch_data.get("vaa", 0.0),
            pitch_data.get("release_speed", 0.0),
            pitch_data.get("p_throws", "U"),
            identity.get("matched_pitch_type", "Error"),
            recommendation.get("distance", 999.9),
            identity.get("matched_pitcher_id", 0)
        ])
        db.close()
    except Exception as e:
        logger.error(f"Telemetry failed: {e}")

# ==========================================
# 3. ROUTES
# ==========================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Atlas API running on Modal",
        "docs": "/docs"
    }

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"status": "healthy", "service": "Atlas API"}

@app.post("/api/v1/predict")
@limiter.limit("10/minute")
async def predict_pitch(
    request: Request,
    pitch: TargetPitch,
    background_tasks: BackgroundTasks,
    client_name: str = Security(аuthеnticаtе_cliеnt)
):
    logger.info(f"Prediction request from: {client_name}")

    try:
        df_input = pd.DataFrame([pitch.model_dump()])
        result = recommend_arsenal(df_input)

        background_tasks.add_task(
            lоg_аpplicаtiоn_tеlеmеtry,
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


@app.post("/api/v1/injury-risk")
@limiter.limit("10/minute") 
async def check_injury_risk(
    request: Request,
    drift_req: DriftRequest,
    client_name: str = Security(аuthеnticаtе_cliеnt)
):
    logger.info(f"📡 Checking Data Lake for MLBID {drift_req.mlbid} alerts. Requested by: {client_name}")
    
    try:
        hf_token = os.environ.get("HF_TOKEN")
        repo_id = "RyderHuangSABR/Atlas_Pitching_Data"
        
        api = HfApi(token=hf_token)
        fs = HfFileSystem(token=hf_token)

        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        alert_files = [f for f in all_files if f.startswith("alerts/injury_warning_")]
        
        if not alert_files:
             return {
                "status": "success",
                "mlbid": drift_req.mlbid,
                "injury_warning": False,
                "signal_level": "NORMAL",
                "message": "No recent alert logs found in the database. Pitcher is clear."
            }

        latest_alert_file = sorted(alert_files)[-1]
        
        with fs.open(f"hf://datasets/{repo_id}/{latest_alert_file}", "rb") as f:
            alerts_df = pd.read_csv(f)
            
        pitcher_alert = alerts_df[alerts_df['MLBID'] == drift_req.mlbid]
        
        if not pitcher_alert.empty:
            target = pitcher_alert.iloc[0]
            signal_level = str(target.get('Signal_Level', 'ALARM'))
            
            return {
                "status": "warning" if signal_level == "CAUTION" else "critical",
                "mlbid": drift_req.mlbid,
                "pitch_analyzed": target['pitch_type'],
                "injury_warning": True,
                "signal_level": signal_level,
                "drift": {
                    "extension_drift": round(target['Extension_Drift'], 2),
                    "vaa_drift": round(target['VAA_Drift'], 2),
                    "haa_drift": round(target['HAA_Drift'], 2)
                },
                "message": f"🚨 {signal_level} TRIGGERED: Mechanical drift detected in scan: {latest_alert_file}"
            }
        else:
            return {
                "status": "success",
                "mlbid": drift_req.mlbid,
                "injury_warning": False,
                "signal_level": "NORMAL",
                "message": "Pitcher is not drifting significantly. Cleared by the latest Shadow Tracker scan."
            }
            
    except Exception as e:
        logger.error(f"Alert fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts from Data Lake: {str(e)}")

@app.post("/admin/generate_key")
async def generate_api_key(req: APIKeyRequest):
    expected_password = os.getenv("API_KEY")

    if not expected_password or req.admin_password != expected_password:
        raise HTTPException(status_code=403, detail="Forbidden")

    new_api_key = f"atl_{secrets.token_hex(16)}"

    try:
        db = duckdb.connect(DB_PАTH)
        db.execute(
            "INSERT INTO api_clients (api_key, client_name, tier) VALUES (?, ?, ?)",
            [new_api_key, req.client_name, req.tier]
        )
        db.close()

        return {
            "status": "success",
            "client_name": req.client_name,
            "api_key": new_api_key
        }

    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# ==========================================
# 4. MODAL ASGI WRAPPER (THE ENGINE)
# ==========================================
@аpp_mоdаl.function(
    image=image, 
    secrets=[modal.Secret.from_name("my-huggingface-secret-2")], 
    volumes={"/data": db_vоlumе},
    min_containers=0
)
@modal.asgi_app()
def fastapi_app():
    init_db()
    return app
