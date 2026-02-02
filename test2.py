import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def debug_first_pair(data_dir, seq_len=512):
    print(f"--- DEBUGGING DATASET LOADING ({data_dir}) ---")
    
    # 1. Find a pair
    search_pattern = os.path.join(data_dir, "**", "imu_*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    sim_path = None
    real_path = None
    
    for f in files:
        if "real_" in os.path.basename(f): continue
        fid = os.path.basename(f).replace("imu_", "").replace(".csv", "")
        r_path = os.path.join(os.path.dirname(f), f"real_imu_{fid}.csv")
        if os.path.exists(r_path):
            sim_path = f
            real_path = r_path
            break
    
    if not sim_path:
        print("❌ No matched pairs found!")
        return

    print(f"Testing Pair:\n  Sim:  {sim_path}\n  Real: {real_path}")

    # 2. Load Data
    try:
        sim_df = pd.read_csv(sim_path)
        real_df = pd.read_csv(real_path)
        print(f"✅ Loaded CSVs. Sim rows: {len(sim_df)}, Real rows: {len(real_df)}")
    except Exception as e:
        print(f"❌ Failed to read CSVs: {e}")
        return

    # 3. Check Columns
    req_sim = ['time', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'imu_ax', 'imu_ay', 'imu_az']
    missing_sim = [c for c in req_sim if c not in sim_df.columns]
    if missing_sim:
        print(f"❌ Missing Sim Columns: {missing_sim}")
        return
    else:
        print("✅ Sim Columns OK")

    # 4. Check Time Alignment
    sim_t = sim_df['time'].values
    sim_t = sim_t - sim_t[0]
    
    if 'time' in real_df.columns:
        real_t = real_df['time'].values
    elif 'time_sec' in real_df.columns:
        real_t = real_df['time_sec'].values
    else:
        print(f"❌ Real file missing 'time' or 'time_sec' column. Found: {real_df.columns.tolist()}")
        return

    print(f"  Raw Real Time[0]: {real_t[0]}")
    real_t = real_t - real_t[0] # Normalize
    
    print(f"  Sim Duration: {sim_t[-1]:.2f}s")
    print(f"  Real Duration: {real_t[-1]:.2f}s")

    # 5. Check Overlap
    valid_mask = real_t <= sim_t[-1]
    real_t_clipped = real_t[valid_mask]
    
    print(f"  Points surviving time alignment: {len(real_t_clipped)} / {len(real_t)}")

    if len(real_t_clipped) < seq_len:
        print(f"❌ FAILURE: Resulting data length ({len(real_t_clipped)}) is smaller than seq_len ({seq_len}).")
        print("   -> SOLUTION: Decrease 'seq_len' in CONFIG or use longer log files.")
        return

    print("✅ SUCCESS: This file would load correctly!")

if __name__ == "__main__":
    # Point this to your actual logs folder
    debug_first_pair('./logs_2')