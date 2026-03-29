import os
import xgboost as xgb
import gc # Garbage Collection
from huggingface_hub import hf_hub_download
from engine.features import PITCH_GROUPS

MODEL_REPO = "RyderHuangSABR/Atlas_Pitching_ML"
_CURRENT_MODEL = {"name": None, "A": None, "B": None}

def get_models_for_pitch(statcast_code):
    global _CURRENT_MODEL
    group_name = PITCH_GROUPS.get(statcast_code)
    
    # If the model is already in RAM, don't reload
    if _CURRENT_MODEL["name"] == group_name:
        return _CURRENT_MODEL
    
    # Clear old model from RAM to stay under 512MB
    _CURRENT_MODEL = {"name": None, "A": None, "B": None}
    gc.collect() 

    try:
        HF_TOKEN = os.getenv("HF_TOKEN")
        path_a = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_A_Whiff_{group_name}.json", token=HF_TOKEN)
        path_b = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_B_Contact_{group_name}.json", token=HF_TOKEN)
        
        model_a = xgb.Booster()
        model_b = xgb.Booster()
        model_a.load_model(path_a)
        model_b.load_model(path_b)
        
        _CURRENT_MODEL = {"name": group_name, "A": model_a, "B": model_b}
        return _CURRENT_MODEL
    except Exception as e:
        print(f"RAM Limit Guard: Error loading {group_name}: {e}")
        return None
