import os
import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta
from huggingface_hub import HfApi, login

print("🕵️‍♂️ Waking up the Scraper...")

# 1. AUTHENTICATE
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN missing! Check GitHub Secrets.")
login(token=HF_TOKEN)

DATA_REPO = "RyderHuangSABR/Atlas_Pitching_Data"

# 2. GET YESTERDAY'S DATE
# Using a 1-day timedelta to grab exactly what happened yesterday
yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
print(f"📅 Pulling MLB raw Statcast data for: {yesterday}")

try:
    # 3. SCRAPE THE DATA
    df = statcast(start_dt=yesterday, end_dt=yesterday)
    
    if df is None or df.empty:
        print("🌙 No games yesterday (Off-season, rainout, or All-Star break). Going back to sleep.")
        exit(0)
        
    print(f"✅ Successfully scraped {len(df)} pitches.")

    # 4. SAVE AND UPLOAD
    file_name = "Yesterday_Pitches.parquet"
    df.to_parquet(file_name, engine='pyarrow')

    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=f"Atlas/{file_name}",
        repo_id=DATA_REPO,
        repo_type="dataset"
    )
    print(f"🚀 Beamed {file_name} straight into the Hugging Face Vault.")

except Exception as e:
    print(f"❌ Scraping failed: {e}")
    exit(1)
