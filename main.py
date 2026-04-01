import os
import secrets
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client

# --- THE BLAST SHIELD IMPORTS ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Your proprietary mathematical engine
from engine.recommender import recommend_arsenal

# 1. Initialize the Rate Limiter (Tracks by API Key, falls back to IP)
def get_api_key_from_request(request: Request):
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(key_func=get_api_key_from_request)

# 2. Initialize the API
app = FastAPI(title="Atlas Pitching Engine", version="4.0-Data-Broker")

# Tell FastAPI what to do when someone breaks the speed limit
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. SUPABASE SETUP (The Cloud VIP List)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("⚠️ WARNING: Supabase keys missing. Bouncer is flying blind.")

api_key_header = APIKeyHeader(name="X-API-Key")

def get_client_identity(api_key: str = Security(api_key_header)):
    """The Bouncer: Checks Supabase for the API Key."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection severed.")
        
    response = supabase.table("api_clients").select("*").eq("api_key", api_key).execute()
    
    if len(response.data) == 0:
        print(f"🚨 BOUNCER ALERT: Rejected invalid key attempt.")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")
    
    client_name = response.data[0]["client_name"]
    return client_name

# 5. Data Models
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

# 6. THE MAIN VIP ENDPOINT (Now with Rate Limiting!)
@app.post("/predict")
@limiter.limit("10/minute") # <--- THE SHIELD. CHANGE THIS STRING TO EXPAND LIMITS.
async def predict(request: Request, pitch: TargetPitch, client_name: str = Security(get_client_identity)):
    
    print(f"⚙️ {client_name} requested a DuckDB scan. Limit: OK.")
    
    target_df = pd.DataFrame([pitch.model_dump()])
    result = recommend_arsenal(target_df)
    
    return {
        "status": "success",
        "client_billed": client_name,
        "prediction": result
    }

# 7. GOD MODE (The Minting Press)
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
        
        print(f"💎 SUCCESS: Minted new {req.tier} key for {req.client_name}")
        
        return {
            "status": "success",
            "message": "Key injected into Supabase vault.",
            "client_name": req.client_name,
            "api_key": new_api_key
        }
    except Exception as e:
        print(f"Supabase Error: {e}")
        raise HTTPException(status_code=500, detail="Database fault during minting.")

# 8. Health Check
@app.get("/")
@limiter.limit("5/minute") # Stops people from spam-pinging the home page
async def health_check(request: Request):
    return {"message": "Atlas API is Live. The bouncer is checking IDs."}
