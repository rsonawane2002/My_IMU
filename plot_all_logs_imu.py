import argparse
import glob
import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


SIM_COLS = ['time', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
REAL_COLS = ['time_sec', 'acc_x_g', 'acc_y_g', 'acc_z_g', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']


def find_log_pair(traj_dir: str) -> Optional[Tuple[str, str]]:
    """Return (sim_path, real_path) inside a traj directory if present."""
    sim_candidates = sorted(glob.glob(os.path.join(traj_dir, 'imu_*.csv')))
    real_candidates = sorted(glob.glob(os.path.join(traj_dir, 'real_imu_*.csv')))
    if not sim_candidates or not real_candidates:
        return None
    # Prefer matching suffix if possible
    for s in sim_candidates:
        suffix = os.path.basename(s).split('_', 1)[-1]  # e.g., '1.csv'
        for r in real_candidates:
            if r.endswith(suffix):
                return s, r
    # Fallback to first pair
    return sim_candidates[0], real_candidates[0]


def load_and_prepare(sim_path: str, real_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim_df = pd.read_csv(sim_path)
    real_df = pd.read_csv(real_path)
    sim_df.columns = sim_df.columns.str.strip()
    real_df.columns = real_df.columns.str.strip()

    # Coerce numeric types where relevant
    for col in SIM_COLS:
        if col in sim_df:
            sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
    for col in REAL_COLS:
        if col in real_df:
            real_df[col] = pd.to_numeric(real_df[col], errors='coerce')

    # Drop rows with missing time
    sim_df = sim_df.dropna(subset=[SIM_COLS[0]])
    real_df = real_df.dropna(subset=[REAL_COLS[0]])

    # Normalize time to start at 0
    sim_df = sim_df.copy()
    real_df = real_df.copy()
    sim_df['t_norm'] = sim_df[SIM_COLS[0]] - sim_df[SIM_COLS[0]].iloc[0]
    real_df['t_norm'] = real_df[REAL_COLS[0]] - real_df[REAL_COLS[0]].iloc[0]
    return sim_df, real_df


def plot_pair(sim_df: pd.DataFrame, real_df: pd.DataFrame, title: str, out_path: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()

    # Accelerometer
    for i, (col_sim, col_real, label) in enumerate(zip(SIM_COLS[1:4], REAL_COLS[1:4], ['ax', 'ay', 'az'])):
        if col_sim in sim_df and col_real in real_df:
            axes[i].plot(sim_df['t_norm'], sim_df[col_sim], label='Sim ' + label)
            axes[i].plot(real_df['t_norm'], real_df[col_real], label='Real ' + label)
        axes[i].set_title(f'IMU {label.upper()}')
        axes[i].set_xlabel('Time [s]')
        axes[i].set_ylabel('Acceleration [g]')
        axes[i].grid(True)
        axes[i].legend(loc='best')

    # Gyroscope
    for i, (col_sim, col_real, label) in enumerate(zip(SIM_COLS[4:], REAL_COLS[4:], ['gx', 'gy', 'gz'])):
        ax = axes[i + 3]
        if col_sim in sim_df and col_real in real_df:
            ax.plot(sim_df['t_norm'], sim_df[col_sim], label='Sim ' + label)
            ax.plot(real_df['t_norm'], real_df[col_real], label='Real ' + label)
        ax.set_title(f'IMU {label.upper()}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angular Velocity [rad/s]')
        ax.grid(True)
        ax.legend(loc='best')

    fig.suptitle(title)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot IMU sim vs real for all logs/traj_* directories.')
    parser.add_argument('--logs_root', default='logs', help='Root containing traj_* subdirectories')
    parser.add_argument('--out', default='runs/log_plots', help='Output directory for plots')
    args = parser.parse_args()

    traj_dirs = sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d))
    if not traj_dirs:
        print(f'No traj_* directories found under {args.logs_root}')
        return

    for traj in traj_dirs:
        pair = find_log_pair(traj)
        if not pair:
            print(f'Skipping {traj}: missing imu_*.csv or real_imu_*.csv')
            continue
        sim_path, real_path = pair
        try:
            sim_df, real_df = load_and_prepare(sim_path, real_path)
        except Exception as e:
            print(f'Failed to load {traj}: {e}')
            continue

        base = os.path.basename(traj)
        out_path = os.path.join(args.out, f'{base}.png')
        title = f'{base}: {os.path.basename(sim_path)} vs {os.path.basename(real_path)}'
        try:
            plot_pair(sim_df, real_df, title, out_path)
            print(f'Wrote {out_path}')
        except Exception as e:
            print(f'Failed to plot {traj}: {e}')


if __name__ == '__main__':
    main()
