import os
import io
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from huggingface_hub import HfApi

print("🚀 Initiating Daily Statcast Fetch (T-Minus 1 Day)...")

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN environment variable is not set. Upload will fail.")

HF_REPO_ID = "RyderHuangSABR/Atlas_Pitching_Data" 

COLUMNS_TO_KEEP = [
    'game_date', 'game_type', 'pitcher', 'pitch_name', 'pitch_type', 
    'release_speed', 'effective_speed', 'release_pos_x', 'release_pos_z', 
    'release_extension', 'pfx_x', 'pfx_z', 'spin_axis', 'release_spin_rate', 
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'description', 'events', 'type' ,'launch_speed','launch_angle',
    'delta_run_exp','estimated_woba_using_speedangle'
]

def fetch_daily_data(target_date_str, is_milb=False):
    league_flag = "&minors=true" if is_milb else ""
    league_name = "MiLB" if is_milb else "MLB"
    
    print(f"📡 Sweeping {league_name} for {target_date_str}...", end=" ")
    
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C&player_type=pitcher&game_date_gt={target_date_str}&game_date_lt={target_date_str}&type=details{league_flag}"
    
    try:
        response = requests.get(url, timeout=30)
        
        if len(response.text) > 500: 
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            
            if not df.empty:
                existing_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
                df = df[existing_cols]
                
                if 'pitcher' in df.columns:
                    df = df.rename(columns={'pitcher': 'MLBID'})
                
                print(f"✅ Secured {len(df):,} pitches.")
                return df
            else:
                print("⚠️ Empty data frame. (No games played)")
        else:
            print("💤 No Data Returned.")
            
    except Exception as e:
        print(f"❌ Server error: {e}")
        
    return pd.DataFrame()

# ==========================================
# THE DAILY ENGINE
# ==========================================

# 1. Lock to US Eastern Time (Prevents UTC rollover bugs)
eastern = ZoneInfo("America/New_York")
yesterday = (datetime.now(eastern) - timedelta(days=1)).strftime('%Y-%m-%d')
daily_pitches = []

# 2. Grab MLB & MiLB
mlb_df = fetch_daily_data(yesterday, is_milb=False)
milb_df = fetch_daily_data(yesterday, is_milb=True)

if not mlb_df.empty: 
    daily_pitches.append(mlb_df)
if not milb_df.empty: 
    daily_pitches.append(milb_df)

# 3. STITCH AND UPLOAD
if daily_pitches:
    print("\n🧬 Stitching the Daily Vault together...")
    daily_df = pd.concat(daily_pitches, ignore_index=True)
    daily_df = daily_df.drop_duplicates()
    
    print(f"🎉 SUCCESS! Total Pitches Secured: {len(daily_df):,}")
    
    # Format the file name dynamically
    file_name = f"Pitches_{yesterday}.parquet"
    
    # Requires 'pyarrow' installed!
    daily_df.to_parquet(file_name, index=False)
    print(f"💾 Saved locally as: {file_name}")
    
    if HF_TOKEN:
        print("🚀 Beaming directly to Hugging Face 'daily_pulls' folder...")
        api = HfApi(token=HF_TOKEN)
        try:
            api.upload_file(
                path_or_fileobj=file_name,
                path_in_repo=f"daily_pulls/{file_name}", 
                repo_id=HF_REPO_ID,
                repo_type="dataset"
            )
            print("🏆 DAILY ARCHITECTURE COMPLETE.")
        except Exception as e:
            print(f"❌ Hugging Face Upload Failed: {e}")
    else:
        print("⚠️ Skipped Hugging Face upload (Missing HF_TOKEN).")
        
else:
    print(f"🌙 No data collected for {yesterday}. Standby mode.")
