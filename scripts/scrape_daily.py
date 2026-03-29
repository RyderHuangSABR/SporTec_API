import os
import io
import requests
import pandas as pd
from datetime import datetime, timedelta
from huggingface_hub import HfApi, login

print("🕵️‍♂️ Booting the Mozilla-Bypass Scraper...")

# ==========================================
# 1. AUTHENTICATE
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN missing! Check GitHub Secrets.")
login(token=HF_TOKEN)

DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"

# ==========================================
# 2. THE MOZILLA ENGINE
# ==========================================
def scrape_savant_csv(start_date, end_date, is_milb=False):
    """
    Directly hits the Baseball Savant backend by mimicking a Mozilla browser.
    """
    level = "MiLB" if is_milb else "MLB"
    print(f"📡 Requesting {level} Data for {start_date}...")

    # The exact backend URL Baseball Savant uses to generate CSVs
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    
    # We pass the exact parameters the website uses, flipping 'minors=true' for MiLB
    params = {
        "all": "true", "hfGT": "R|", "player_type": "pitcher",
        "game_date_gt": start_date, "game_date_lt": end_date,
        "minors": "true" if is_milb else "false", 
        "type": "details"
    }
    
    # 🚨 THE BYPASS: We tell the server we are a real human on a Mozilla browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/csv"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=60)
        
        # If Savant catches us or throws an error, we catch it
        if response.status_code != 200:
            print(f"⚠️ Warning: Savant returned status {response.status_code} for {level}.")
            return pd.DataFrame()
            
        # Decode the raw CSV text straight into Pandas memory
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        if df.empty or 'pitch_type' not in df.columns:
            return pd.DataFrame()
            
        # Tag it so we know where it came from
        df['league_level'] = level
        return df

    except Exception as e:
        print(f"❌ Error pulling {level}: {e}")
        return pd.DataFrame()

# ==========================================
# 3. EXECUTE THE HEIST
# ==========================================
# Look at yesterday's date
yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

# 1. Grab Major Leagues
mlb_df = scrape_savant_csv(yesterday, yesterday, is_milb=False)

# 2. Grab Minor Leagues
milb_df = scrape_savant_csv(yesterday, yesterday, is_milb=True)

# Combine them into one massive dataset
combined_df = pd.concat([mlb_df, milb_df], ignore_index=True)

if combined_df.empty:
    print("🌙 No games yesterday (Off-season or rainouts). Going back to sleep.")
    exit(0)

print(f"✅ Successfully extracted {len(combined_df)} total pitches (MLB + MiLB).")

# ==========================================
# 4. UPLOAD TO THE VAULT
# ==========================================
file_name = "Yesterday_Pitches.parquet"
combined_df.to_parquet(file_name, engine='pyarrow')

api = HfApi()
api.upload_file(
    path_or_fileobj=file_name,
    path_in_repo=f"Atlas/{file_name}",
    repo_id=DATA_REPO,
    repo_type="dataset"
)

print("🚀 Payload delivered to Hugging Face Vault. The Engine is ready to process.")
