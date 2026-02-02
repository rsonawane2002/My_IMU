import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


REAL_COLS = ['time', 'acc_x_g', 'acc_y_g', 'acc_z_g', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']


def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in REAL_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['time'])
    # Normalize time
    df = df.copy()
    df['t_norm'] = df['time'] - df['time'].iloc[0]
    return df


def compute_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 8:
        return np.array([]), np.array([])
    x = np.asarray(x) - np.mean(x)
    nperseg = min(4096, max(256, (len(x)//8)//2*2))  # even, reasonable
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2, detrend='constant', scaling='density')
    return f, Pxx


def top_peaks(f: np.ndarray, p: np.ndarray, k: int = 3, fmin: float = 0.5, fmax: float = None) -> List[Tuple[float, float]]:
    if fmax is None:
        fmax = f[-1] if len(f) else 0.0
    if len(f) == 0:
        return []
    mask = (f >= fmin) & (f <= fmax)
    f2, p2 = f[mask], p[mask]
    if len(f2) == 0:
        return []
    idx = np.argpartition(p2, -k)[-k:]
    idx = idx[np.argsort(-p2[idx])]  # sort descending
    return [(float(f2[i]), float(p2[i])) for i in idx]


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Overlay PSDs of real IMU across trajectories and report dominant peaks.')
    ap.add_argument('--logs_root', default='log_temp')
    ap.add_argument('--out', default='runs/psd_plots')
    ap.add_argument('--crop_start_s', type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    traj_dirs = sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d))
    real_paths = [sorted(glob.glob(os.path.join(d, 'real_imu_*.csv')))[0] for d in traj_dirs if glob.glob(os.path.join(d, 'real_imu_*.csv'))]
    if not real_paths:
        print('No real_imu_*.csv found under', args.logs_root)
        return

    # Colors per trajectory
    colors = plt.cm.tab10.colors

    # Prepare figures
    fig_acc, axes_acc = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig_gyr, axes_gyr = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    peak_report: Dict[str, List[Tuple[str, List[Tuple[float, float]]]]] = {'acc': [], 'gyr': []}

    for idx, rp in enumerate(real_paths):
        name = os.path.basename(os.path.dirname(rp))  # traj_x
        df = load_real(rp)
        if args.crop_start_s > 0:
            df = df[df['t_norm'] >= args.crop_start_s].reset_index(drop=True)
        t = df['t_norm'].to_numpy()
        if len(t) < 2:
            continue
        fs = 1.0 / float(np.median(np.diff(t)))

        # Accelerometer
        for i, col in enumerate(['acc_x_g', 'acc_y_g', 'acc_z_g']):
            f, P = compute_psd(df[col].to_numpy(), fs)
            if len(f) == 0:
                continue
            axes_acc[i].plot(f, P, color=colors[idx % len(colors)], alpha=0.7, label=name)
            if i == 0:
                peak_report['acc'].append((name, top_peaks(f, P, k=3, fmin=1.0, fmax=min(200.0, f[-1]))))

        # Gyroscope
        for i, col in enumerate(['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']):
            f, P = compute_psd(df[col].to_numpy(), fs)
            if len(f) == 0:
                continue
            axes_gyr[i].plot(f, P, color=colors[idx % len(colors)], alpha=0.7, label=name)
            if i == 0:
                peak_report['gyr'].append((name, top_peaks(f, P, k=3, fmin=1.0, fmax=min(200.0, f[-1]))))

    for i, ax in enumerate(axes_acc):
        ax.set_title(['Accel X', 'Accel Y', 'Accel Z'][i])
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [units^2/Hz]')
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    for i, ax in enumerate(axes_gyr):
        ax.set_title(['Gyro X', 'Gyro Y', 'Gyro Z'][i])
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [units^2/Hz]')
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig_acc.suptitle('Real IMU Accel PSDs across trajectories')
    fig_gyr.suptitle('Real IMU Gyro PSDs across trajectories')

    fig_acc.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_gyr.tight_layout(rect=[0, 0.03, 1, 0.95])

    acc_path = os.path.join(args.out, 'real_acc_psd_overlay.png')
    gyr_path = os.path.join(args.out, 'real_gyro_psd_overlay.png')
    fig_acc.savefig(acc_path, dpi=150)
    fig_gyr.savefig(gyr_path, dpi=150)
    plt.close(fig_acc)
    plt.close(fig_gyr)

    print('Wrote:', acc_path)
    print('Wrote:', gyr_path)

    # Print a compact peak summary for X axes as a proxy; others visible in plots.
    print('Dominant peaks (approx., X-axes) [Hz, PSD]:')
    for k in ('acc', 'gyr'):
        for name, peaks in peak_report[k]:
            print(f'  {name} {k}:', ', '.join(f"{f:.1f}Hz" for f, _ in peaks))


if __name__ == '__main__':
    main()

