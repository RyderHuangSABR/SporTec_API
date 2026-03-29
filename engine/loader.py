import os
import xgboost as xgb
from huggingface_hub import hf_hub_download
from engine.features import PITCH_GROUPS

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "RyderHuangSABR/Atlas_Pitching_ML"

_MODEL_CACHE = {}

def load_all_models():
    global _MODEL_CACHE
    if _MODEL_CACHE: return _MODEL_CACHE
    unique_groups = set(PITCH_GROUPS.values())
    for group_name in unique_groups:
        try:
            path_a = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_A_Whiff_{group_name}.json", token=HF_TOKEN)
            path_b = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_B_Contact_{group_name}.json", token=HF_TOKEN)
            model_a, model_b = xgb.Booster(), xgb.Booster()
            model_a.load_model(path_a)
            model_b.load_model(path_b)
            _MODEL_CACHE[group_name] = {"A": model_a, "B": model_b}
        except Exception as e:
            print(f"Skipping {group_name}: {e}")
    return _MODEL_CACHE

def get_models_for_pitch(statcast_code):
    models = load_all_models()
    group_name = PITCH_GROUPS.get(statcast_code)
    return models.get(group_name)
