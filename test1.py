import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def compute_metrics(sim_arr, real_arr):
    """
    Computes correlation, scale (slope), and bias (intercept).
    """
    if len(sim_arr) < 2 or len(real_arr) < 2:
        return 0.0, 1.0, 0.0
        
    # Correlation
    corr = np.corrcoef(sim_arr, real_arr)[0, 1]
    if np.isnan(corr): corr = 0.0
    
    # Linear Regression (Real = Scale * Sim + Bias)
    try:
        slope, intercept = np.polyfit(sim_arr, real_arr, 1)
    except:
        slope, intercept = 1.0, 0.0
        
    return corr, slope, intercept

def get_column_data(df, potential_names):
    """
    Searches the dataframe for the first matching column name from the list.
    """
    for name in potential_names:
        if name in df.columns:
            return df[name].to_numpy()
    return None

def diagnose_log(sim_csv, real_csv):
    # 1. Load Data
    try:
        df_sim = pd.read_csv(sim_csv)
        df_real = pd.read_csv(real_csv)
        
        # vital: strip whitespace from headers
        df_sim.columns = df_sim.columns.str.strip()
        df_real.columns = df_real.columns.str.strip()
    except Exception as e:
        # print(f"  [Error reading] {e}")
        return None, None

    # 2. Check for required Sim columns based on your headers
    # We strictly require these to exist, otherwise we skip the file
    required_sim_cols = ['time', 'acc_x', 'ang_vel_x']
    if not all(col in df_sim.columns for col in required_sim_cols):
        # Silently skip files that don't match the format (e.g. processed/aligned intermediates)
        return None, None

    # 3. Time Alignment
    t_sim = df_sim['time'].to_numpy()
    
    # Real logs often use 'time_sec' or 'time'
    t_real = get_column_data(df_real, ['time_sec', 'time', 't'])
    
    if t_real is None:
        print(f"  [Skip] {os.path.basename(real_csv)} missing time column")
        return None, None

    # Normalize Start Times
    t_sim = t_sim - t_sim[0]
    t_real = t_real - t_real[0]

    # Crop to overlapping duration
    if t_sim[-1] < 0.1 or t_real[-1] < 0.1:
        return None, None

    t_max = min(t_sim[-1], t_real[-1])
    mask = t_real <= t_max
    t_real = t_real[mask]
    df_real = df_real.iloc[:len(t_real)]

    # 4. Interpolate Sim Data to Real Time Grid
    # Mappings based on YOUR headers:
    sim_map = {
        'ax': 'acc_x', 'ay': 'acc_y', 'az': 'acc_z',
        'gx': 'ang_vel_x', 'gy': 'ang_vel_y', 'gz': 'ang_vel_z'
    }
    
    sim_interp = {}
    for axis_key, col_name in sim_map.items():
        if col_name not in df_sim.columns:
            return None, None
        
        val_sim = df_sim[col_name].to_numpy()
        f = interp1d(t_sim, val_sim, fill_value="extrapolate")
        sim_interp[axis_key] = f(t_real)

    # 5. Diagnostic Loop
    # We assume Real data uses standard names, but we look for alternatives just in case
    real_map_candidates = {
        'ax': ['acc_x_g', 'acc_x', 'ax'],
        'ay': ['acc_y_g', 'acc_y', 'ay'],
        'az': ['acc_z_g', 'acc_z', 'az'],
        'gx': ['gyro_x_rad_s', 'gyro_x', 'gx', 'ang_vel_x'],
        'gy': ['gyro_y_rad_s', 'gyro_y', 'gy', 'ang_vel_y'],
        'gz': ['gyro_z_rad_s', 'gyro_z', 'gz', 'ang_vel_z'],
    }

    results_gyro = []
    results_acc = []

    # Check GYRO (gx, gy, gz)
    for axis in ['gx', 'gy', 'gz']:
        r_vals = get_column_data(df_real, real_map_candidates[axis])
        if r_vals is None: continue

        best_corr = 0
        best_match = "None"
        best_scale = 1.0

        # Try matching Real Axis against ALL Sim Axes to check for swaps
        for sim_axis in ['gx', 'gy', 'gz']:
            s_vals = sim_interp[sim_axis]
            corr, scale, bias = compute_metrics(s_vals, r_vals)
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_match = sim_axis
                best_scale = scale
        
        # Format: "GX matches gy (C: 0.99, S: 1.0)"
        results_gyro.append(f"{axis.upper()}->{best_match} (C:{best_corr:.2f}, S:{best_scale:.2f})")

    # Check ACCEL (ax, ay, az)
    for axis in ['ax', 'ay', 'az']:
        r_vals = get_column_data(df_real, real_map_candidates[axis])
        if r_vals is None: continue

        best_corr = 0
        best_match = "None"
        best_bias = 0.0

        for sim_axis in ['ax', 'ay', 'az']:
            s_vals = sim_interp[sim_axis]
            corr, scale, bias = compute_metrics(s_vals, r_vals)
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_match = sim_axis
                best_bias = bias

        results_acc.append(f"{axis.upper()}->{best_match} (C:{best_corr:.2f}, B:{best_bias:.2f})")

    return results_gyro, results_acc

def main():
    logs_root = 'logs'
    # Only grab files ending in _with_gravity.csv
    files = sorted(glob.glob(os.path.join(logs_root, 'traj_*', '*_with_gravity.csv')))
    
    print(f"{'LOG NAME':<12} | {'TYPE':<5} | {'MATCH RESULTS'}")
    print("-" * 120)

    count = 0
    for sim_path in files:
        # Filter out intermediate files if they exist to keep output clean
        if '_processed_' in sim_path or '_aligned_' in sim_path:
            continue
            
        folder = os.path.dirname(sim_path)
        base = os.path.basename(sim_path).replace('_with_gravity.csv', '')

        # Find corresponding real log
        real_files = glob.glob(os.path.join(folder, 'real_imu_*.csv'))
        if not real_files:
            continue
            
        real_path = real_files[0]
        
        # Run Diagnosis
        gyr_res, acc_res = diagnose_log(sim_path, real_path)
        
        if gyr_res and acc_res:
            print(f"{base[:12]:<12} | GYRO  | {', '.join(gyr_res)}")
            print(f"{'':<12} | ACCEL | {', '.join(acc_res)}")
            print("-" * 120)
            count += 1

    if count == 0:
        print("No valid comparisons found. Check your file paths or column names.")

if __name__ == '__main__':
    main()