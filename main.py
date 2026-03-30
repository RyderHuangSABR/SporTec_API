import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Your proprietary mathematical engine (Safe in the kitchen)
from engine.recommender import recommend_arsenal

# 1. Initialize the API
app = FastAPI(title="Atlas Pitching Engine", version="3.0-Enterprise")

# 2. CORS Middleware: Allows external UIs to talk to your API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. THE VIP GUEST LIST (The API Keys)
# You generate these random strings and email them to the teams.
# They put this string in their code. They NEVER see your code.
AUTHORIZED_CLIENTS = {
    # The Master Key (Set this in Render Environment Variables)
    os.getenv("ATLAS_MASTER_KEY", "ryder_master_001"): "Atlas_Admin",
    
    # Client 1: The Giants ($0 - Partnership)
    "atl_7f8a9b2c4d1e3f5a6b": "San Francisco Giants",
    
    # Client 2: A Wall Street Hedge Fund ($250k/year)
    "atl_99zz88yy77xx66ww55": "Cohen Point72 Sports",
    
    # Client 3: USA Baseball Olympic Team
    "atl_olympic_gold_2028": "USA Baseball"
}

api_key_header = APIKeyHeader(name="X-API-Key")

def get_client_identity(api_key: str = Security(api_key_header)):
    """The Bouncer: Checks the ID and returns the name on the list."""
    if api_key not in AUTHORIZED_CLIENTS:
        print(f"🚨 BOUNCER ALERT: Rejected invalid key attempt: {api_key[:5]}...")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")
    
    client_name = AUTHORIZED_CLIENTS[api_key]
    print(f"🔑 ACCESS GRANTED: {client_name} has entered the server.")
    return client_name

# 4. Define the exact JSON structure they must send you
class TargetPitch(BaseModel):
    vaa: float
    haa: float
    release_extension: float
    spin_axis: float
    release_speed: float
    pfx_x: float
    pfx_z: float

# 5. The Main VIP Endpoint
@app.post("/predict")
async def predict(pitch: TargetPitch, client_name: str = Security(get_client_identity)):
    
    # We log who is making the request so you can bill them later
    print(f"⚙️ Running DuckDB Euclidean Math for: {client_name}")
    
    # Convert their JSON into a DataFrame for your engine
    target_df = pd.DataFrame([pitch.dict()])
    
    # Run your memory-optimized recommendation engine
    result = recommend_arsenal(target_df)
    
    # Send ONLY the final answer back to them
    return {
        "status": "success",
        "client_billed": client_name,
        "prediction": result
    }

# 6. A public health-check (No API key required just to see if the server is awake)
@app.get("/")
async def health_check():
    return {"message": "Atlas API is Live. The bouncer is checking IDs."}
