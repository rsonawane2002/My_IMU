import pandas as pd
import numpy as np
import os
import sys

# Ensure python can find your existing physics engine
sys.path.append(os.getcwd()) 
from imu_sim2real_plus.sensors.imu_synth import synth_measurements

# --- CONFIGURATION (Datasheet Defaults) ---
INPUT_FILE = "/home/sameer/Documents/imu_sim2real_plus_v2/clean_stationary_24h.csv"
OUTPUT_FILE = "/home/sameer/Documents/imu_sim2real_plus_v2/noisy_stationary_24h.csv"

def run_datasheet_physics(df_clean):
    print("Preparing data...")
    t = df_clean['time'].to_numpy()
    
    # FORCE 104Hz dt to match Datasheet ODR
    dt = 1.0 / 104.0 
    
    N = len(t)
    # Check headers - adjust if your clean csv has different names
    # Assuming standard Isaac names: imu_ax, imu_ay... or acc_x, acc_y...
    # Let's try standardizing
    if 'imu_ax' in df_clean.columns:
        f_B_input = df_clean[['imu_ax', 'imu_ay', 'imu_az']].to_numpy()
        w_B_input = df_clean[['imu_gx', 'imu_gy', 'imu_gz']].to_numpy()
    elif 'acc_x' in df_clean.columns:
        f_B_input = df_clean[['acc_x', 'acc_y', 'acc_z']].to_numpy()
        w_B_input = df_clean[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].to_numpy()
    else:
        raise ValueError("Could not find standard IMU columns in input CSV.")

    # Mock Kinematics
    R_BW = np.repeat(np.eye(3)[np.newaxis, :, :], N, axis=0) 
    g_W = np.array([0.0, 0.0, 9.80665])
    a_W = f_B_input + g_W 
    wdot_B = np.gradient(w_B_input, dt, axis=0)
    r_lever = np.zeros(3)

    # --- DATASHEET CONFIG ---
    # Matches the YAML you provided (2g range, low noise, NO vibration)
    cfg = {
        'imu': {
            'quantization_bits': 16, 
            'accel_fs_g': 2.0,       # ±2 g
            'gyro_fs_dps': 250.0,    # ±250 dps
            'misalignment_pct': [-1.0, 1.0], 
            'accel': {
                'scale_ppm': [-3000, 3000],
                'bias_init': [-0.00981, 0.00981], 
                'bias_tau_s': [200, 2000],
                'noise_density': [0.0005884, 0.0005884] # 60 ug/rtHz
            },
            'gyro': {
                'scale_ppm': [-1000, 1000],
                'bias_init': [-0.001, 0.001],
                'bias_tau_s': [200, 2000],
                'noise_density': [8.7266e-05, 8.7266e-05] # 5 mdps/rtHz
            }
        },
        'vibration': None # Intentionally disabled for Allan Variance
    }

    print("Running Physics Engine (This may take a moment for 24h data)...")
    # Zero RPM since vibration is off
    rpm_profile = np.zeros_like(t)

    sim_out, _ = synth_measurements(
        R_WB=R_BW, 
        w_B=w_B_input,
        a_W=a_W,
        wdot_B=wdot_B,
        r_lever=r_lever,
        cfg=cfg,
        dt=dt,
        seed=42, 
        rpm_profile=rpm_profile
    )

    print("Packaging output...")
    out = pd.DataFrame({
        'time': t,
        'imu_ax': sim_out['f_meas'][:, 0],
        'imu_ay': sim_out['f_meas'][:, 1],
        'imu_az': sim_out['f_meas'][:, 2],
        'imu_gx': sim_out['w_meas'][:, 0],
        'imu_gy': sim_out['w_meas'][:, 1],
        'imu_gz': sim_out['w_meas'][:, 2],
    })
    return out

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        exit(1)

    df_clean = pd.read_csv(INPUT_FILE)
    df_noisy = run_datasheet_physics(df_clean)
    
    df_noisy.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Noisy data saved to: {OUTPUT_FILE}")