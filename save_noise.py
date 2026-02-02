import os
import glob
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imu_vibration_sim import (
    ResonantMode,
    simulate_imu_with_vibration,
)

# Reuse helpers from plotting script for consistency
from plot_all_logs_imu import (
    load_and_prepare,
    plot_pair,
    compute_axis_alignment_separate,
    apply_axis_alignment_separate,
)


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def load_clean(clean_csv: str) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)
    df.columns = df.columns.str.strip()
    req = [
        'time','acc_x','acc_y','acc_z',
        'ang_vel_x','ang_vel_y','ang_vel_z',
        'quat_w','quat_x','quat_y','quat_z',
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{os.path.basename(clean_csv)} missing columns: {miss}")
    df = df.copy()
    df['t_norm'] = df['time'] - df['time'].iloc[0]
    return df


def synth_with_motor_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed version with MUCH HIGHER gain parameters to match real vibration amplitudes.
    """
    t = df['time'].to_numpy()
    
    # Build specific force f_B from clean log: f_B = a_B - g_B
    a_B_input = df[['acc_x','acc_y','acc_z']].to_numpy()
    w_B = df[['ang_vel_x','ang_vel_y','ang_vel_z']].to_numpy()
    quats = df[['quat_w','quat_x','quat_y','quat_z']].to_numpy()
    R_WB = np.array([quaternion_to_rotation_matrix(q) for q in quats])
    g_W = np.array([0.0, 0.0, 9.81])
    g_B = np.einsum('nij,j->ni', R_WB, g_W)
    f_B_true = a_B_input - g_B

    # CRITICAL FIX: Gain parameters increased by ~50-100x to match real vibration RMS
    # Real data shows RMS of 0.7-1.7 after high-pass filtering
    # Previous gains were producing RMS of only 0.025
    
    modes_acc = [
        # Low-frequency (10-50 Hz) - Real data shows 16-22% energy here
        ResonantMode(f0=16.7,  zeta=0.05, gain=3.0, axes=(0,1,2)),   # Peak in traj_118
        ResonantMode(f0=35.0,  zeta=0.05, gain=2.0, axes=(0,1,2)),
        
        # Mid-frequency motor harmonics (50-150 Hz) - 18-45% energy
        ResonantMode(f0=75.0,  zeta=0.03, gain=4.0, axes=(0,1,2)),
        ResonantMode(f0=95.0,  zeta=0.03, gain=3.0, axes=(0,1,2)),
        ResonantMode(f0=108.3, zeta=0.03, gain=3.5, axes=(0,1,2)),   # Peak in traj_118
        ResonantMode(f0=120.0, zeta=0.03, gain=5.0, axes=(0,1,2)),
        ResonantMode(f0=125.0, zeta=0.03, gain=4.5, axes=(0,1,2)),   # Peak in traj_134
        ResonantMode(f0=133.3, zeta=0.03, gain=3.5, axes=(1,)),      # Peak in traj_147
        
        # High-frequency structural resonances (150-300 Hz) - 27-49% energy!
        # THIS IS WHERE MOST ENERGY IS - needs HIGHEST gains
        ResonantMode(f0=180.0, zeta=0.04, gain=6.0, axes=(0,1,2)),
        ResonantMode(f0=208.3, zeta=0.04, gain=8.0, axes=(0,1,2)),   # Major peak in traj_137
        ResonantMode(f0=233.3, zeta=0.04, gain=9.0, axes=(0,1,2)),   # Major peak in traj_118, traj_134
        ResonantMode(f0=250.0, zeta=0.05, gain=7.0, axes=(2,)),      # Strong on Z-axis
        ResonantMode(f0=280.0, zeta=0.05, gain=5.0, axes=(0,1,2)),
        
        # Very high frequency (300-500 Hz) - 2.5-19% energy
        ResonantMode(f0=325.0, zeta=0.06, gain=3.0, axes=(0,1,2)),   # Peak in traj_118
        ResonantMode(f0=400.0, zeta=0.07, gain=2.0, axes=(0,1,2)),
        ResonantMode(f0=450.0, zeta=0.08, gain=1.5, axes=(0,1,2)),   # Peak in traj_118
    ]
    
    modes_gyr = [
        # Gyro modes need adjustment too, but they're less critical
        ResonantMode(f0=6.5,   zeta=0.07, gain=0.08, axes=(0,1,2)),  # 4x increase
        ResonantMode(f0=8.0,   zeta=0.08, gain=0.06, axes=(0,1,2)),  # 4x increase
        ResonantMode(f0=16.0,  zeta=0.06, gain=0.05, axes=(0,1,2)),
        ResonantMode(f0=75.0,  zeta=0.03, gain=0.015, axes=(2,)),
        ResonantMode(f0=120.0, zeta=0.03, gain=0.015, axes=(2,)),
    ]

    # Motor speed profile
    rpm = np.full_like(t, 75.0*60.0)  # 4500 rpm constant

    w_meas, a_meas, _info = simulate_imu_with_vibration(
        t, true_w_B=w_B, true_a_B=f_B_true, rpm_profile=rpm,
        modes_accel=modes_acc, modes_gyro=modes_gyr,
        accel_white_std=0.02, gyro_white_std=0.0025,
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
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    x = x - np.mean(x); y = y - np.mean(y)
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0:
        return float('nan')
    return float(np.corrcoef(x/sx, y/sy)[0,1])


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Inject motor-like vibration into clean logs and compare to real IMU')
    ap.add_argument('--logs_root', default='logs')
    ap.add_argument('--out', default='runs/vib_compare')
    ap.add_argument('--crop_start_s', type=float, default=0.4)
    ap.add_argument('--save_noisy', action='store_true', 
                    help='Save noisy synthetic IMU data to trajectory folders')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    traj_dirs = sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d))
    summary_rows: List[List[str]] = [['traj','corr_ax','corr_ay','corr_az','corr_gx','corr_gy','corr_gz']]

    for traj in traj_dirs:
        clean_glob = sorted(glob.glob(os.path.join(traj, 'clean_*.csv')))
        real_glob  = sorted(glob.glob(os.path.join(traj, 'real_imu_*.csv')))
        if not clean_glob or not real_glob:
            print('Skipping', traj, '(missing clean or real)')
            continue
        clean_path = clean_glob[0]
        real_path  = real_glob[0]
        base = os.path.basename(traj)

        # Load and synthesize
        clean_df = load_clean(clean_path)
        synth_df = synth_with_motor_bands(clean_df)

        # ============ NEW: Save the noisy synthetic data ============
        if args.save_noisy:
            # Extract trajectory number from folder name (e.g., traj_255 -> 255)
            traj_num = base.split('_')[-1]
            noisy_output_path = os.path.join(traj, f'synth_imu_{traj_num}.csv')
            
            # Save with column names matching real IMU format for easier comparison
            synth_save_df = pd.DataFrame({
                'time_sec': synth_df['time'],
                'acc_x_g': synth_df['imu_ax'],
                'acc_y_g': synth_df['imu_ay'],
                'acc_z_g': synth_df['imu_az'],
                'gyro_x_rad_s': synth_df['imu_gx'],
                'gyro_y_rad_s': synth_df['imu_gy'],
                'gyro_z_rad_s': synth_df['imu_gz'],
            })
            synth_save_df.to_csv(noisy_output_path, index=False)
            print(f'  → Saved noisy synthetic data: {noisy_output_path}')
        # ============================================================

        # Load real for plotting/comparison
        real_df = pd.read_csv(real_path)
        real_df.columns = real_df.columns.str.strip()
        real_df['t_norm'] = real_df['time_sec'] - real_df['time_sec'].iloc[0]

        # Optional crop first so alignment emphasizes the region of interest
        if args.crop_start_s > 0:
            synth_df = synth_df[synth_df['t_norm'] >= args.crop_start_s].reset_index(drop=True)
            real_df  = real_df[ real_df['t_norm']  >= args.crop_start_s].reset_index(drop=True)

        # Compute best signed-permutation alignment matrices (acc and gyro
        # independently) using correlations to the real log.
        try:
            A_acc, A_gyr = compute_axis_alignment_separate(synth_df, real_df)
            synth_df_al = apply_axis_alignment_separate(synth_df, A_acc, A_gyr)
        except Exception as e:
            # Fall back to unaligned if something goes wrong
            print(f'Axis alignment failed for {base}: {e}; using identity map')
            A_acc = np.eye(3); A_gyr = np.eye(3)
            synth_df_al = synth_df

        # Plot overlay
        out_path = os.path.join(args.out, f'{base}_vib_vs_real.png')
        plot_pair(synth_df_al, real_df, f'{base}: vib-synth vs real (auto-aligned, per-sensor)', out_path)
        print('Wrote', out_path)

        # Persist the alignment matrices for reference/repro
        np.savetxt(os.path.join(args.out, f'{base}_axis_alignment_acc.csv'), A_acc, fmt='%.6f', delimiter=',')
        np.savetxt(os.path.join(args.out, f'{base}_axis_alignment_gyr.csv'), A_gyr, fmt='%.6f', delimiter=',')

        # Correlations (interpolate synth to real time grid)
        t_ref = real_df['t_norm'].to_numpy()
        t_src = synth_df['t_norm'].to_numpy()
        def interp(col):
            return np.interp(t_ref, t_src, synth_df[col].to_numpy())
        metrics: Dict[str, float] = {}
        for s_col, r_col, name in [
            ('imu_ax','acc_x_g','ax'),
            ('imu_ay','acc_y_g','ay'),
            ('imu_az','acc_z_g','az'),
            ('imu_gx','gyro_x_rad_s','gx'),
            ('imu_gy','gyro_y_rad_s','gy'),
            ('imu_gz','gyro_z_rad_s','gz'),
        ]:
            s_al = np.interp(t_ref, synth_df_al['t_norm'].to_numpy(), synth_df_al[s_col].to_numpy())
            r = real_df[r_col].to_numpy()
            metrics[f'corr_{name}'] = corrcoef(s_al, r)

        summary_rows.append([
            base,
            *(f"{metrics[k]:.6f}" for k in ['corr_ax','corr_ay','corr_az','corr_gx','corr_gy','corr_gz'])
        ])

    # Save summary
    out_csv = os.path.join(args.out, 'correlations.csv')
    with open(out_csv, 'w') as f:
        for row in summary_rows:
            f.write(','.join(row) + '\n')
    print('Wrote', out_csv)


if __name__ == '__main__':
    main()