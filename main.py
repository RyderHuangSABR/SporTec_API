import os
import secrets
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client

# Your proprietary mathematical engine (Safe in the kitchen)
from engine.recommender import recommend_arsenal

# 1. Initialize the API
app = FastAPI(title="Atlas Pitching Engine", version="4.0-Data-Broker")

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. SUPABASE SETUP (The Cloud VIP List)
# Add these to your Render Environment Variables!
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Only initialize Supabase if the keys are present (prevents local crashing)
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
        
    # Query Supabase for the exact key
    response = supabase.table("api_clients").select("*").eq("api_key", api_key).execute()
    
    if len(response.data) == 0:
        print(f"🚨 BOUNCER ALERT: Rejected invalid key attempt: {api_key[:8]}...")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")
    
    # Grab the client's name from the database response
    client_name = response.data[0]["client_name"]
    print(f"🔑 ACCESS GRANTED: {client_name} has entered the server.")
    
    return client_name

# 4. Data Models (What clients MUST send you)
class TargetPitch(BaseModel):
    p_throws: str  # 'R' or 'L' - Enforces handedness quarantine
    vaa: float
    haa: float
    release_extension: float
    spin_axis: float
    release_speed: float
    pfx_x: float
    pfx_z: float

class MintRequest(BaseModel):
    client_name: str
    tier: str
    admin_password: str

# 5. THE MAIN VIP ENDPOINT (The Kitchen)
@app.post("/predict")
async def predict(pitch: TargetPitch, client_name: str = Security(get_client_identity)):
    
    print(f"⚙️ Running DuckDB Euclidean Math for: {client_name}")
    
    # Convert their JSON into a DataFrame
    target_df = pd.DataFrame([pitch.dict()])
    
    # Run your memory-optimized recommendation engine
    result = recommend_arsenal(target_df)
    
    return {
        "status": "success",
        "client_billed": client_name,
        "prediction": result
    }

# 6. GOD MODE (The Minting Press)
@app.post("/admin/mint_key")
async def mint_api_key(req: MintRequest):
    """Generates a new API key and saves it to Supabase instantly."""
    
    # Check your Master Password (Set in Render Env Vars)
    expected_password = os.getenv("ATLAS_MASTER_KEY", "ryder_admin_override_99")
    if req.admin_password != expected_password:
        print("🚨 ALERT: Unauthorized access to God Mode.")
        raise HTTPException(status_code=403, detail="Nice try. Access Denied.")
    
    # Forge the new cryptographic key
    new_api_key = f"atl_{secrets.token_hex(16)}"
    
    # Inject it directly into your free Supabase database
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

# 7. Health Check
@app.get("/")
async def health_check():
    return {"message": "Atlas API is Live. The bouncer is checking IDs."}
