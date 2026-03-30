import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pandas as pd

# Import your engine functions
from engine.loader import load_all_models
from engine.recommender import recommend_arsenal

# 1. Initialize the API
app = FastAPI(title="Atlas Pitching Engine", version="1.0")

# 2. Setup the Bouncer (API Key Security)
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    # Grab the secret password from Render's secure vault
    expected_key = os.getenv("ATLAS_SECRET_KEY")
    
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server Error: Missing Secret Key.")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Get off my server.")
    return api_key

# 3. Define the exact JSON structure your iPad will send
class TargetPitch(BaseModel):
    vaa: float
    haa: float
    release_extension: float
    spin_axis: float
    release_speed: float
    pfx_x: float
    pfx_z: float

# 4. Pre-load the models when the server boots up
@app.on_event("startup")
async def startup_event():
    print("🚀 Booting up Atlas Engine...")
    load_all_models()
    print("✅ XGBoost Models loaded.")

# 5. The Main VIP Endpoint
@app.post("/predict")
async def predict(pitch: TargetPitch, key: str = Security(get_api_key)):
    # Convert the iPad's JSON into a single-row Pandas DataFrame
    target_df = pd.DataFrame([pitch.dict()])
    
    # Run your memory-optimized recommendation engine
    result = recommend_arsenal(target_df)
    
    return {
        "status": "success",
        "prediction": result
    }

# 6. A public health-check (No password required, just to prove it's awake)
@app.get("/")
async def health_check():
    return {"message": "Atlas API is Live. The bouncer is at the door."}
