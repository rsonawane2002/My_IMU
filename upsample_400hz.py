import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import glob
from tqdm import tqdm

def normalize_in_place(data_dir, target_hz=400.0):
    dt = 1.0 / target_hz
    # Recursively find all simulation IMU files
    sim_files = glob.glob(os.path.join(data_dir, "**", "imu_*.csv"), recursive=True)

    print(f"Normalizing all data to a fixed {target_hz}Hz in-place...")
    for sim_path in tqdm(sim_files):
        filename = os.path.basename(sim_path)
        folder = os.path.dirname(sim_path)
        
        # Skip files that are already 'real_' or already processed (400hz)
        if filename.startswith("real_") or "400hz" in filename:
            continue
            
        file_id = filename.replace("imu_", "").replace(".csv", "")
        real_path = os.path.join(folder, f"real_imu_{file_id}.csv")
        
        if not os.path.exists(real_path):
            continue

        # Load Data
        df_s = pd.read_csv(sim_path)
        df_r = pd.read_csv(real_path)

        # 1. Zero out time and find common duration
        df_s['time'] = df_s['time'] - df_s['time'].iloc[0]
        df_r['time'] = df_r['time'] - df_r['time'].iloc[0]
        t_max = min(df_s['time'].iloc[-1], df_r['time'].iloc[-1])
        
        # 2. Create the perfect 400Hz time grid
        new_time = np.arange(0, t_max, dt)

        # 3. Upsample Simulation (e.g., 60Hz -> 400Hz)
        s_cols = [c for c in df_s.columns if c != 'time']
        s_data = {'time': new_time}
        for col in s_cols:
            f = interp1d(df_s['time'], df_s[col], kind='linear', fill_value="extrapolate")
            s_data[col] = f(new_time)
        df_s_fixed = pd.DataFrame(s_data)

        # 4. Resample Real (High Jitter -> 400Hz Stable)
        r_cols = [c for c in df_r.columns if c != 'time']
        r_data = {'time': new_time}
        for col in r_cols:
            f = interp1d(df_r['time'], df_r[col], kind='linear', fill_value="extrapolate")
            r_data[col] = f(new_time)
        df_r_fixed = pd.DataFrame(r_data)

        # 5. Save in the same subfolder with new names
        sim_save_path = os.path.join(folder, f"imu_400hz_{file_id}.csv")
        real_save_path = os.path.join(folder, f"real_400hz_{file_id}.csv")
        
        df_s_fixed.to_csv(sim_save_path, index=False)
        df_r_fixed.to_csv(real_save_path, index=False)

# Run the normalization
normalize_in_place('./new_pipeline', target_hz=400.0)