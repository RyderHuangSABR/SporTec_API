# engine/loader.py
import os
import logging
import xgboost as xgb
from huggingface_hub import hf_hub_download
from engine.features import PITCH_GROUPS

logger = logging.getLogger(__name__)
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "RyderHuangSABR/Atlas_Pitching_ML"

_MODEL_CACHE = {}

def load_atlas_data():
    """Pulls the master historical file and the player dictionary."""
    logger.info("Fetching Master Data and Dictionary...")
    
    # 1. Fetch the master pitching data
    data_path = hf_hub_download(
        repo_id="RyderHuangSABR/Atlas_Pitching_Data", 
        filename="Atlas_Pitching.parquet", 
        repo_type="dataset", # Specify it's a dataset repo
        token=HF_TOKEN
    )
    df_master = pd.read_parquet(data_path)
    
    # 2. Fetch the player dictionary
    dict_path = hf_hub_download(
        repo_id="RyderHuangSABR/Atlas_Pitching_Data", 
        filename="MLB_Player_Dictionary.parquet", 
        repo_type="dataset",
        token=HF_TOKEN
    )
    df_dict = pd.read_parquet(dict_path)
    
    return df_master, df_dict

def get_models_for_pitch(statcast_code: str):
    """Retrieves and caches XGBoost models from Hugging Face for a given pitch type."""
    global _MODEL_CACHE
    group_name = PITCH_GROUPS.get(statcast_code)
    
    if not group_name: 
        return None

    if group_name in _MODEL_CACHE:
        return _MODEL_CACHE[group_name]

    logger.info(f"Loading models into cache for pitch group: {group_name}")
    try:
        path_a = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_A_Whiff_{group_name}.json", token=HF_TOKEN)
        path_b = hf_hub_download(repo_id=MODEL_REPO, filename=f"Engine_B_Contact_{group_name}.json", token=HF_TOKEN)
        
        model_a, model_b = xgb.Booster(), xgb.Booster()
        model_a.load_model(path_a)
        model_b.load_model(path_b)
        
        _MODEL_CACHE[group_name] = {"A": model_a, "B": model_b}
        return _MODEL_CACHE[group_name]
    
    except Exception as e:
        logger.error(f"Failed to load models for {group_name}: {e}")
        return None
