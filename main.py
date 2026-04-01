import os
import secrets
from fastapi import FastAPI, HTTPException, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.recommender import recommend_arsenal

def get_api_key_from_request(request: Request):
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(key_func=get_api_key_from_request)

app = FastAPI(title="Atlas Pitching Engine", version="4.0-Data-Broker")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("⚠️ WARNING: Supabase keys missing.")

api_key_header = APIKeyHeader(name="X-API-Key")

def get_client_identity(api_key: str = Security(api_key_header)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection severed.")
        
    response = supabase.table("api_clients").select("*").eq("api_key", api_key).execute()
    
    if len(response.data) == 0:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")
    
    return response.data[0]["client_name"]

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
    """Silently harvests the API request data into Supabase to expand the 1-NN memory."""
    if supabase:
        try:
            telemetry_payload = {
                "client_name": client_name,
                "input_vaa": pitch_data["vaa"],
                "input_speed": pitch_data["release_speed"],
                "input_p_throws": pitch_data["p_throws"],
                "recommended_pitch": recommendation.get("recommended_pitch", "Error"),
                "kinematic_distance": recommendation.get("kinematic_distance", 999.9),
                "apex_clone_mlbid": recommendation.get("apex_clone_mlbid", 0)
            }
            supabase.table("telemetry_logs").insert(telemetry_payload).execute()
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
    
    # Send the data to Supabase in the background
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
        new_client_data = {
            "api_key": new_api_key,
            "client_name": req.client_name,
            "tier": req.tier
        }
        supabase.table("api_clients").insert(new_client_data).execute()
        
        return {
            "status": "success",
            "message": "Key injected into Supabase vault.",
            "client_name": req.client_name,
            "api_key": new_api_key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database fault during minting.")

@app.get("/")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"message": "Atlas 1-NN Engine is Live."}
