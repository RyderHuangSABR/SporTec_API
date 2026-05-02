# engine/loader.py
import os
import logging
import xgboost as xgb
import pandas as pd
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
    # 🚨 FIX: Changed filename to include the "Atlas/" folder path
    data_path = hf_hub_download(
        repo_id="RyderHuangSABR/Atlas_Pitching_Data", 
        filename="Atlas/Atlas_Pitching.parquet", # Added the folder path here
        repo_type="dataset",
        token=HF_TOKEN
    )
    df_master = pd.read_parquet(data_path)
    
    # 2. Fetch the player dictionary
    # 🚨 NOTE: Check if this file is also in a folder! 
    # If it is in the same folder, change it to "Atlas/MLB_Player_Dictionary.parquet"
    dict_path = hf_hub_download(
        repo_id="RyderHuangSABR/Atlas_Pitching_Data", 
        filename="MLB_Player_Dictionary.parquet", 
        repo_type="dataset",
        token=HF_TOKEN
    )
    df_dict = pd.read_parquet(dict_path)
    
    return df_master, df_dict

# ... rest of your get_models_for_pitch function remains the same
