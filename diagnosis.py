import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import joblib
from scipy.interpolate import interp1d
from scipy import signal

# Load your CONFIG from the main script or redefine here
DATA_DIR = './logs_2'
SCALER_PATH = 'scaler.pkl'

def diagnose_pipeline():
    print(f"--- Starting Pipeline Diagnosis ---")
    
    # 1. Check for Scaler
    if not os.path.exists(SCALER_PATH):
        print("[CRITICAL] Scaler file not found. Did you delete it for the 400Hz run?")
    else:
        scaler = joblib.load(SCALER_PATH)
        print(f"[INFO] Scaler loaded. Features expected: {scaler.n_features_in_}")

    # 2. Get Files (Same logic as your main script)
    import glob
    search_pattern = os.path.join(DATA_DIR, "**", "imu_400hz_*.csv")
    sim_files = sorted(glob.glob(search_pattern, recursive=True))
    
    issues = []

    for sim_path in tqdm(sim_files, desc="Checking Pairs"):
        file_id = os.path.basename(sim_path).replace("imu_400hz_", "").replace(".csv", "")
        real_path = os.path.join(os.path.dirname(sim_path), f"real_400hz_{file_id}.csv")
        
        if not os.path.exists(real_path):
            continue

        try:
            # --- TEST 1: RAW DATA INTEGRITY ---
            s_df = pd.read_csv(sim_path)
            r_df = pd.read_csv(real_path)

            for name, df in [("Sim", s_df), ("Real", r_df)]:
                if df.isnull().values.any():
                    issues.append(f"NaN found in raw CSV: {name} - {file_id}")
                if np.isinf(df.select_dtypes(include=[np.number])).values.any():
                    issues.append(f"Inf found in raw CSV: {name} - {file_id}")

            # --- TEST 2: ALIGNMENT MATH ---
            # This is where 400Hz upsampling usually fails
            sim_time = s_df['time'].values - s_df['time'].values[0]
            real_time = r_df['time'].values - r_df['time'].values[0]
            
            # Check for zero-delta time (Interpolation Killer)
            if (np.diff(sim_time) <= 0).any() or (np.diff(real_time) <= 0).any():
                issues.append(f"Non-increasing timestamps in {file_id}")

            # Test Gyro Norm for Alignment
            # (Adjust column names here if they differ in your 400Hz files!)
            sim_gyro = s_df[['imu_gx', 'imu_gy', 'imu_gz']].values
            real_gyro = r_df[['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
            
            s_mag = np.linalg.norm(sim_gyro, axis=1)
            r_mag = np.linalg.norm(real_gyro, axis=1)

            if np.max(s_mag) == 0 or np.max(r_mag) == 0:
                issues.append(f"Flat/Zero Gyro data in {file_id} (Alignment will fail)")

        except Exception as e:
            issues.append(f"Crash processing {file_id}: {str(e)}")

    print(f"\n--- Diagnosis Results ---")
    if not issues:
        print("Success: No obvious data corruption found. The issue likely lies in the Model's Log-Var layer.")
    else:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:20]: # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20: print(f"  ... and {len(issues)-20} more.")

if __name__ == "__main__":
    diagnose_pipeline()