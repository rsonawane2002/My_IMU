import pandas as pd
import numpy as np
import glob
import os

def verify_400hz_alignment(data_dir):
    # Find the newly created 400hz files
    sim_files = glob.glob(os.path.join(data_dir, "**", "imu_400hz_*.csv"), recursive=True)
    
    if not sim_files:
        print("No 400Hz files found. Check your file paths or the previous script output.")
        return

    print(f"{'File ID':<10} | {'Hz (Sim)':<10} | {'Hz (Real)':<10} | {'Aligned?':<10} | {'Jitter (ms)':<12}")
    print("-" * 65)

    for sim_path in sim_files:
        folder = os.path.dirname(sim_path)
        filename = os.path.basename(sim_path)
        file_id = filename.replace("imu_400hz_", "").replace(".csv", "")
        real_path = os.path.join(folder, f"real_400hz_{file_id}.csv")

        if not os.path.exists(real_path):
            continue

        # Load timestamps
        df_s = pd.read_csv(sim_path, usecols=['time'])
        df_r = pd.read_csv(real_path, usecols=['time'])

        # Calculate DTs
        dt_s = np.diff(df_s['time'].values)
        dt_r = np.diff(df_r['time'].values)

        # Calculate Frequencies
        hz_s = 1.0 / np.mean(dt_s) if len(dt_s) > 0 else 0
        hz_r = 1.0 / np.mean(dt_r) if len(dt_r) > 0 else 0
        
        # Check for perfect temporal alignment (Are the timestamps identical?)
        is_aligned = np.allclose(df_s['time'].values, df_r['time'].values, atol=1e-7)
        
        # Calculate Jitter (Standard Deviation of DT)
        jitter_ms = np.std(dt_r) * 1000

        print(f"{file_id:<10} | {hz_s:<10.2f} | {hz_r:<10.2f} | {str(is_aligned):<10} | {jitter_ms:<12.6f}")

verify_400hz_alignment('./logs_2')