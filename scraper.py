import os
import io
import requests
import pandas as pd
from datetime import datetime, timedelta
from huggingface_hub import HfApi

print("🕵️‍♂️ Booting the Atlas Automated Scraper...")

# ==========================================
# 1. AUTHENTICATE THE VAULT
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN missing! The GitHub Action did not pass the secret.")

DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"

# ==========================================
# 2. THE MOZILLA-BYPASS ENGINE
# ==========================================
def scrape_savant_csv(start_date, end_date, is_milb=False):
    level = "MiLB" if is_milb else "MLB"
    print(f"📡 Requesting {level} Data for {start_date}...")

    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    
    params = {
        "all": "true", "hfGT": "R|", "player_type": "pitcher",
        "game_date_gt": start_date, "game_date_lt": end_date,
        "minors": "true" if is_milb else "false", 
        "type": "details"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/csv"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=60)
        
        if response.status_code != 200:
            print(f"⚠️ Warning: Savant returned status {response.status_code} for {level}.")
            return pd.DataFrame()
            
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        if df.empty or 'pitch_type' not in df.columns:
            return pd.DataFrame()
            
        df['league_level'] = level
        return df

    except Exception as e:
        print(f"❌ Error pulling {level}: {e}")
        return pd.DataFrame()

# ==========================================
# 3. EXECUTE THE DAILY HEIST
# ==========================================
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

mlb_df = scrape_savant_csv(yesterday, yesterday, is_milb=False)
milb_df = scrape_savant_csv(yesterday, yesterday, is_milb=True)

# Safely combine only the dataframes that actually have data
frames = [df for df in [mlb_df, milb_df] if not df.empty]

if not frames:
    print("🌙 No valid pitches found for yesterday. Engine returning to standby mode.")
    exit(0)

combined_df = pd.concat(frames, ignore_index=True)
print(f"✅ Successfully extracted {len(combined_df)} total pitches (MLB + MiLB).")

# ==========================================
# 4. COMPRESS AND UPLOAD
# ==========================================
file_name = f"Pitches_{yesterday}.parquet"
combined_df.to_parquet(file_name, engine='pyarrow')

print(f"📦 Compressing into Parquet format: {file_name}")

# Pass the token DIRECTLY into the HfApi to avoid CI/CD cache permission crashes
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=file_name,
    path_in_repo=f"daily_pulls/{file_name}", 
    repo_id=DATA_REPO,
    repo_type="dataset"
)

print("🚀 Payload delivered to Hugging Face Vault. The pipeline is complete.")
