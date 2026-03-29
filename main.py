import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from engine.loader import load_all_models, get_models_for_pitch
from engine.scoring import score_pitch
from engine.features import FEATURES
from engine.recommender import recommend_arsenal

app = FastAPI(title="Atlas Engine API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("🚀 Booting Atlas Cloud Server...")
    load_all_models()
    print("✅ Models loaded. System Ready.")

class PitchSimulationRequest(BaseModel):
    pitch_type: str
    release_speed: float
    pfx_x: float
    pfx_z: float
    release_extension: float
    release_pos_x: float
    release_pos_z: float
    spin_axis: float
    release_spin_rate: float
    vy0: float
    ay: float
    vz0: float
    az: float
    vx0: float
    ax: float

@app.post("/api/v1/simulate")
async def simulate_custom_pitch(req: PitchSimulationRequest):
    try:
        df = pd.DataFrame([req.dict()])
        
        # Hawk-Eye Physics
        radicand = df['vy0']**2 - 2 * df['ay'] * (50 - 17/12)
        radicand = np.clip(radicand, a_min=0, a_max=None) 
        t = (-df['vy0'] - np.sqrt(radicand)) / df['ay']
        
        vy_f = df['vy0'] + df['ay'] * t
        vz_f = df['vz0'] + df['az'] * t
        vx_f = df['vx0'] + df['ax'] * t
        
        df['vaa'] = np.rad2deg(np.arctan(vz_f / vy_f))
        df['haa'] = np.rad2deg(np.arctan(vx_f / vy_f))
        df['reaction_time'] = (50 - df['release_extension']) / abs(df['vy0'])
        
        t_commit = 0.167 
        df['commit_x'] = df['release_pos_x'] + (df['vx0'] * t_commit) + (0.5 * df['ax'] * (t_commit**2))
        df['commit_z'] = df['release_pos_z'] + (df['vz0'] * t_commit) + (0.5 * df['az'] * (t_commit**2))
        df['movement_ratio'] = abs(df['pfx_x']) / (abs(df['pfx_z']) + 0.1)

        # AI Inference
        df['pitch_type'] = df['pitch_type'].astype('category')
        models = get_models_for_pitch(req.pitch_type)
        if not models: raise ValueError(f"No ML model for {req.pitch_type}")
            
        q_plus, c_plus = score_pitch(df, models, FEATURES)

        return {
            "status": 200,
            "data": {
                "quality_plus": round(q_plus, 1),
                "command_plus": round(c_plus, 1),
                "kinematics": {
                    "vaa": round(float(df['vaa'].iloc[0]), 2),
                    "haa": round(float(df['haa'].iloc[0]), 2)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/recommend")
async def get_pitch_recommendation(req: PitchSimulationRequest):
    try:
        target_df = pd.DataFrame([req.dict()])
        
        # Calculate VAA/HAA for the KNN
        radicand = target_df['vy0']**2 - 2 * target_df['ay'] * (50 - 17/12)
        radicand = np.clip(radicand, a_min=0, a_max=None) 
        t = (-target_df['vy0'] - np.sqrt(radicand)) / target_df['ay']
        vz_f = target_df['vz0'] + target_df['az'] * t
        vy_f = target_df['vy0'] + target_df['ay'] * t
        vx_f = target_df['vx0'] + target_df['ax'] * t
        
        target_df['vaa'] = np.rad2deg(np.arctan(vz_f / vy_f))
        target_df['haa'] = np.rad2deg(np.arctan(vx_f / vy_f))
        
        recommendation = recommend_arsenal(target_df)
        return {"status": 200, "data": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
