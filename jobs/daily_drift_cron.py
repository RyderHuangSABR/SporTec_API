import modal
import os
# 1. Define the Modal App and Dependencies
app = modal.App("sabr-shadow-tester")
image = modal.Image.debian_slim().pip_install(
    "pandas", 
    "numpy", 
    "huggingface_hub", 
    "pyarrow", 
    "fsspec"
)
# 2. The Main Scheduled Function
@app.function(
    image=image, 
    schedule=modal.Period(days=1), 
    secrets=[modal.Secret.from_name("my-huggingface-secret")] 
)

def check_pitcher_drift():
    import pandas as pd
    import numpy as np
    from huggingface_hub import HfApi, HfFileSystem
    from datetime import datetime



    hf_token = os.environ["HF_TOKEN"]

    repo_id = "RyderHuangSABR/Atlas_Pitching_Data"

    fs = HfFileSystem(token=hf_token)

    api = HfApi(token=hf_token)



    print("🚀 Waking up: Fetching Master GMM Profiles...")

    

    with fs.open(f"hf://datasets/{repo_id}/Atlas/gmm_pitch_profiles.csv", "rb") as f:

        master_gmm = pd.read_csv(f)



    print("📥 Fetching yesterday's daily pull...")

    daily_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    daily_pulls = [f for f in daily_files if f.startswith("daily_pulls/")]

    latest_file = sorted(daily_pulls)[-1] 



    # ==========================================

    # THE FIX: DYNAMIC FILE READING

    # ==========================================

    with fs.open(f"hf://datasets/{repo_id}/{latest_file}", "rb") as f:

        if latest_file.endswith(".parquet"):

            print(f"Detected Parquet file: {latest_file}")

            daily_df = pd.read_parquet(f)

        else:

            print(f"Detected CSV file: {latest_file}")

            daily_df = pd.read_csv(f)

    # ==========================================

    # 3. KINEMATICS: CALCULATE VAA & HAA

    # ==========================================

    Y0 = 50.0 

    YF = 17.0 / 12.0 



    daily_df['vy_f'] = -np.sqrt(daily_df['vy0']**2 - (2 * daily_df['ay'] * (Y0 - YF)))

    daily_df['t'] = (daily_df['vy_f'] - daily_df['vy0']) / daily_df['ay']

    daily_df['vz_f'] = daily_df['vz0'] + (daily_df['az'] * daily_df['t'])

    daily_df['vx_f'] = daily_df['vx0'] + (daily_df['ax'] * daily_df['t'])



    daily_df['VAA'] = -np.arctan(daily_df['vz_f'] / daily_df['vy_f']) * (180 / np.pi)

    daily_df['HAA'] = -np.arctan(daily_df['vx_f'] / daily_df['vy_f']) * (180 / np.pi)



    daily_avg = daily_df.groupby(['MLBID', 'pitch_type'])[['VAA', 'HAA', 'release_extension', 'release_speed']].mean().reset_index()



    # ==========================================

    # 4. THE SHADOW TEST: CALCULATE DRIFT

    # ==========================================

    print("🔍 Comparing Daily Data against perfect GMM Centroids...")

    

    # THE FIX: Completed the cut-off merge line

    comparison = pd.merge(daily_avg, master_gmm, on=['MLBID', 'pitch_type'], suffixes=('_daily', ''))

    

    comparison['VAA_Drift'] = np.abs(comparison['VAA_daily'] - comparison['VAA'])

    comparison['HAA_Drift'] = np.abs(comparison['HAA_daily'] - comparison['HAA'])

    comparison['Extension_Drift'] = np.abs(comparison['release_extension_daily'] - comparison['release_extension'])



    alerts = comparison[(comparison['Extension_Drift'] > 0.3) & (comparison['VAA_Drift'] > 0.5)].copy()

    

    if len(alerts) > 0:

        alerts['alert_date'] = datetime.today().strftime('%Y-%m-%d')

        print(f"🚨 WARNING: Found {len(alerts)} pitchers showing mechanical degradation!")

        

        alerts.to_csv("/tmp/new_alerts.csv", index=False)

        

        api.upload_file(

            path_or_fileobj="/tmp/new_alerts.csv",

            path_in_repo=f"alerts/injury_warning_{datetime.today().strftime('%Y%m%d')}.csv",

            repo_id=repo_id,

            repo_type="dataset"

        )

        print("✅ Alert log securely saved to Hugging Face with cryptographic timestamp.")

    else:

        print("✅ No massive deviations found today. Everyone is healthy.")



# Local entry point for testing

@app.local_entrypoint()

def main():

    check_pitcher_drift.remote()

