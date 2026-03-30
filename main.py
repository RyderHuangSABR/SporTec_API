import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pandas as pd

# Notice: We ONLY import the recommender now. We don't import the loader at startup anymore!
from engine.recommender import recommend_arsenal

# 1. Initialize the API
app = FastAPI(title="Atlas Pitching Engine", version="2.0-Lite")

# 2. Setup the Bouncer (API Key Security)
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
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

# Notice: The entire @app.on_event("startup") block is COMPLETELY GONE.
# The server now boots up instantly with 0 models loaded in RAM.

# 4. The Main VIP Endpoint
@app.post("/predict")
async def predict(pitch: TargetPitch, key: str = Security(get_api_key)):
    # Convert the iPad's JSON into a single-row Pandas DataFrame
    target_df = pd.DataFrame([pitch.dict()])
    
    # Run your memory-optimized, DuckDB recommendation engine
    result = recommend_arsenal(target_df)
    
    return {
        "status": "success",
        "prediction": result
    }

# 5. A public health-check
@app.get("/")
async def health_check():
    return {"message": "Atlas API is Live. DuckDB Engine Online. The bouncer is at the door."}
