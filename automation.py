import os
import glob
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom imports
from imu_vibration_sim import (
    ResonantMode,
    simulate_imu_with_vibration,
)
from plot_all_logs_imu import (
    load_and_prepare,
    plot_pair,
)

def load_clean(clean_csv: str) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)
    df.columns = df.columns.str.strip()
    
    # Map specific CSV columns to internal variable names
    col_map = {
        'imu_ax': 'acc_x',
        'imu_ay': 'acc_y',
        'imu_az': 'acc_z',
        'imu_gx': 'ang_vel_x',
        'imu_gy': 'ang_vel_y',
        'imu_gz': 'ang_vel_z',
        'time':   'time'
    }
    
    # Validation
    miss = [k for k in col_map.keys() if k not in df.columns]
    if miss:
        raise ValueError(f"{os.path.basename(clean_csv)} missing columns: {miss}")
    
    # Rename columns to standard internal names
    df = df.rename(columns=col_map)
    df['t_norm'] = df['time'] - df['time'].iloc[0]
    return df

def synth_with_motor_bands(df: pd.DataFrame) -> pd.DataFrame:
    t = df['time'].to_numpy()
    
    f_B_true = df[['acc_x','acc_y','acc_z']].to_numpy()
    w_B = df[['ang_vel_x','ang_vel_y','ang_vel_z']].to_numpy()

    # --- NOISE PARAMETERS ---
    modes_acc = [
        ResonantMode(f0=75.0,  zeta=0.02, gain=0.10, axes=(0,1,2)),
        ResonantMode(f0=95.0,  zeta=0.03, gain=0.04, axes=(0,1,2)),
        ResonantMode(f0=120.0, zeta=0.03, gain=0.08, axes=(2,)),
        ResonantMode(f0=130.0, zeta=0.03, gain=0.06, axes=(2,)),
    ]
    modes_gyr = [
        ResonantMode(f0=6.5,   zeta=0.07, gain=0.04, axes=(0,1,2)),
        ResonantMode(f0=8.0,   zeta=0.08, gain=0.03, axes=(0,1,2)),
        ResonantMode(f0=75.0,  zeta=0.03, gain=0.008, axes=(2,)),
        ResonantMode(f0=120.0, zeta=0.03, gain=0.008, axes=(2,)),
    ]

    rpm = np.full_like(t, 75.0*60.0)  # 4500 rpm constant

    w_meas, a_meas, _info = simulate_imu_with_vibration(
        t, true_w_B=w_B, true_a_B=f_B_true, rpm_profile=rpm,
        modes_accel=modes_acc, modes_gyro=modes_gyr,
        accel_white_std=0.04,
        gyro_white_std=0.005,
        gyro_bias_rw_std=2e-5, g_sensitivity=0.002,
        quantize_accel_mg=None, quantize_gyro_dps=None, seed=123)

    out = pd.DataFrame({
        'time': t,
        'imu_ax': a_meas[:,0],
        'imu_ay': a_meas[:,1],
        'imu_az': a_meas[:,2],
        'imu_gx': w_meas[:,0],
        'imu_gy': w_meas[:,1],
        'imu_gz': w_meas[:,2],
    })
    out['t_norm'] = out['time'] - out['time'].iloc[0]
    return out

def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2: return float('nan')
    x = x - np.mean(x); y = y - np.mean(y)
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0: return float('nan')
    return float(np.corrcoef(x/sx, y/sy)[0,1])

def main():
    ap = argparse.ArgumentParser(description='Inject motor-like vibration into clean logs')
    ap.add_argument('--logs_root', default='logs_2')
    ap.add_argument('--out', default='runs/vib_compare')
    ap.add_argument('--crop_start_s', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Get all subdirectories in logs_2 that look like 'traj_*'
    traj_dirs = sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d))
    summary_rows: List[List[str]] = [['traj','corr_ax','corr_ay','corr_az','corr_gx','corr_gy','corr_gz']]

    print(f"Found {len(traj_dirs)} directories in {args.logs_root}...")

    for traj_dir in traj_dirs:
        # 1. EXTRACT ID from folder name (e.g. 'traj_251' -> '251')
        folder_name = os.path.basename(traj_dir) 
        try:
            traj_id = folder_name.split('_')[-1]
        except:
            print(f"[{folder_name}] Could not parse ID, skipping.")
            continue

        # 2. CONSTRUCT FILE PATHS SPECIFICALLY
        # Input: imu_400hz_251.csv
        clean_filename = f'imu_400hz_{traj_id}.csv'
        clean_path = os.path.join(traj_dir, clean_filename)

        # Output: sim_noisy_251.csv
        sim_out_filename = f'sim_noisy_{traj_id}.csv'
        sim_out_path = os.path.join(traj_dir, sim_out_filename)

        # Real data (for comparison): real_400hz_251.csv
        real_filename = f'real_400hz_{traj_id}.csv'
        real_path = os.path.join(traj_dir, real_filename)
        
        # 3. VERIFY INPUT EXISTS
        if not os.path.exists(clean_path):
            print(f"[{folder_name}] Skipping - missing {clean_filename}")
            continue

        # 4. LOAD AND SYNTHESIZE
        try:
            clean_df = load_clean(clean_path)
            synth_df = synth_with_motor_bands(clean_df)
        except Exception as e:
            print(f"[{folder_name}] Error processing: {e}")
            continue

        # 5. SAVE NOISY CSV
        synth_df.to_csv(sim_out_path, index=False)
        print(f"[{folder_name}] Generated: {sim_out_filename}")

        # 6. COMPARE (Only if real data exists)
        if not os.path.exists(real_path):
            # No real data, just move on
            continue

        real_df = pd.read_csv(real_path)
        real_df.columns = real_df.columns.str.strip()
        real_df['t_norm'] = real_df['time'] - real_df['time'].iloc[0]

        if args.crop_start_s > 0:
            synth_df = synth_df[synth_df['t_norm'] >= args.crop_start_s].reset_index(drop=True)
            real_df  = real_df[ real_df['t_norm']  >= args.crop_start_s].reset_index(drop=True)

        synth_df_al = synth_df.copy()

        # Plot overlay
        out_plot_name = f'{folder_name}_vib_vs_real.png'
        out_plot_path = os.path.join(args.out, out_plot_name)
        plot_pair(synth_df_al, real_df, f'{folder_name}: vib-synth vs real', out_plot_path)

        # Correlations 
        t_ref = real_df['t_norm'].to_numpy()
        metrics: Dict[str, float] = {}
        for s_col, r_col, name in [
            ('imu_ax','acc_x_g','ax'), ('imu_ay','acc_y_g','ay'), ('imu_az','acc_z_g','az'),
            ('imu_gx','gyro_x_rad_s','gx'), ('imu_gy','gyro_y_rad_s','gy'), ('imu_gz','gyro_z_rad_s','gz'),
        ]:
            s_al = np.interp(t_ref, synth_df_al['t_norm'].to_numpy(), synth_df_al[s_col].to_numpy())
            r = real_df[r_col].to_numpy()
            metrics[f'corr_{name}'] = corrcoef(s_al, r)

        summary_rows.append([
            folder_name,
            *(f"{metrics[k]:.6f}" for k in ['corr_ax','corr_ay','corr_az','corr_gx','corr_gy','corr_gz'])
        ])

    # Save summary of correlations
    if len(summary_rows) > 1:
        out_csv = os.path.join(args.out, 'correlations.csv')
        with open(out_csv, 'w') as f:
            for row in summary_rows:
                f.write(','.join(row) + '\n')
        print('Wrote correlation summary to', out_csv)

if __name__ == '__main__':
    main()