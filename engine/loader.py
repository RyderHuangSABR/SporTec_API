import os
import xgboost as xgb
from huggingface_hub import hf_hub_download
from engine.features import PITCH_GROUPS

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "RyderHuangSABR/Atlas_Pitching_ML"

_MODEL_CACHE = {}

def get_models_for_pitch(statcast_code):
    global _MODEL_CACHE
    group_name = PITCH_GROUPS.get(statcast_code)
    
    if not group_name: 
        return None

    # If it's already in RAM, use it. If not, load ONLY this one.
    if group_name in _MODEL_CACHE:
        return _MODEL_CACHE[group_name]

    print(f"🧠 Lazy Loading models for {group_name}...")
    try:
        path_a = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_A_Whiff_{group_name}.json", token=HF_TOKEN)
        path_b = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_B_Contact_{group_name}.json", token=HF_TOKEN)
        
        model_a, model_b = xgb.Booster(), xgb.Booster()
        model_a.load_model(path_a)
        model_b.load_model(path_b)
        
        _MODEL_CACHE[group_name] = {"A": model_a, "B": model_b}
        return _MODEL_CACHE[group_name]
    except Exception as e:
        print(f"Error loading {group_name}: {e}")
        return None
