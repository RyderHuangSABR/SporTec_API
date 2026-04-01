import os, secrets, pandas as pd
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.recommender import recommend_arsenal

# 1. SETUP
def get_api_key(request: Request):
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(key_func=get_api_key)
app = FastAPI(title="Atlas Hexcore API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 2. MODELS
class TargetPitch(BaseModel):
    p_throws: str; vaa: float; haa: float; release_extension: float; 
    release_pos_z: float; release_pos_x: float; fastball_speed: float; 
    release_speed: float; spin_axis: float

# 3. THE BOUNCER (Now smarter)
async def verify_and_log(api_key: str, input_data: dict, output_data: dict):
    """The Hexcore learns: Log every request for future training."""
    try:
        # 1. Verify and get client info
        res = supabase.table("api_clients").select("*").eq("api_key", api_key).execute()
        if not res.data:
            raise HTTPException(status_code=401, detail="Invalid Key")
        
        client = res.data[0]
        
        # 2. Log telemetry (This is how your data 'evolves')
        log_entry = {
            "client_id": client['id'],
            "input_json": input_data,
            "prediction": output_data.get("recommended_pitch"),
            "q_plus": output_data.get("expected_quality_plus")
        }
        supabase.table("request_logs").insert(log_entry).execute()
        
        return client
    except:
        raise HTTPException(status_code=401, detail="Access Denied")

# 4. ENDPOINTS
@app.post("/predict")
@limiter.limit("20/minute") # Increased for the 'Glorious Evolution'
async def predict(request: Request, pitch: TargetPitch):
    api_key = request.headers.get("X-API-Key")
    if not api_key: raise HTTPException(status_code=401)

    # Run Math
    input_dict = pitch.model_dump()
    result = recommend_arsenal(pd.DataFrame([input_dict]))
    
    # Verify & Log Telemetry
    await verify_and_log(api_key, input_dict, result)
    
    return result

@app.post("/admin/mint_key")
async def mint_key(client_name: str, tier: str, admin_password: str):
    if admin_password != os.getenv("ATLAS_MASTER_KEY"):
        raise HTTPException(status_code=403)
    
    new_key = f"atl_{secrets.token_hex(16)}"
    supabase.table("api_clients").insert({
        "api_key": new_key, "client_name": client_name, "tier": tier
    }).execute()
    
    return {"key": new_key}

@app.get("/")
async def health(): return {"status": "Glorious Evolution in progress."}
