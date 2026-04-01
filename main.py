import os
import secrets
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.recommender import recommend_arsenal

# --- SMARTER RATE LIMITING ---
def get_api_key_from_request(request: Request):
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(key_func=get_api_key_from_request)
app = FastAPI(title="Atlas Pitching Engine", version="4.5-Glorious-Evolution")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SUPABASE BOUNCER ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

api_key_header = APIKeyHeader(name="X-API-Key")

async def get_client_identity(api_key: str = Security(api_key_header)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database Offline.")
    
    # Smarter query: get client name AND their custom rate limit from Supabase
    res = supabase.table("api_clients").select("client_name, rate_limit").eq("api_key", api_key).execute()
    
    if len(res.data) == 0:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    
    return res.data[0] # Returns a dict: {"client_name": "...", "rate_limit": "10/minute"}

# --- MODELS ---
class TargetPitch(BaseModel):
    p_throws: str; vaa: float; haa: float; release_extension: float
    release_pos_z: float; release_pos_x: float; fastball_speed: float
    release_speed: float; spin_axis: float

class MintRequest(BaseModel):
    client_name: str; tier: str; limit_string: str; admin_password: str

# --- ENDPOINTS ---
@app.post("/predict")
@limiter.limit("10/minute") # Default fallback
async def predict(request: Request, pitch: TargetPitch, client: dict = Security(get_client_identity)):
    # The client['rate_limit'] is now available if you want to do dynamic limiting!
    target_df = pd.DataFrame([pitch.model_dump()])
    result = recommend_arsenal(target_df)
    return {"status": "success", "client": client['client_name'], "prediction": result}

@app.post("/admin/mint_key")
async def mint_api_key(req: MintRequest):
    if req.admin_password != os.getenv("ATLAS_MASTER_KEY"):
        raise HTTPException(status_code=403, detail="Forbidden.")
    
    new_key = f"atl_{secrets.token_hex(16)}"
    new_client_data = {
        "api_key": new_key,
        "client_name": req.client_name,
        "tier": req.tier,
        "rate_limit": req.limit_string # e.g., "50/minute"
    }
    supabase.table("api_clients").insert(new_client_data).execute()
    return {"status": "success", "api_key": new_key}

@app.get("/")
async def health():
    return {"status": "online", "engine": "DuckDB-Hexcore"}
