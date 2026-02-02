import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse helpers from plotting script
from plot_all_logs_imu import (
    find_log_pair,
    find_clean,
    load_and_prepare,
    synth_from_clean,
    apply_fixed_axis_map,
)


def align_to(t_ref: np.ndarray, t_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    """Interpolate y_src sampled at t_src onto t_ref (1D)."""
    # Ensure strictly increasing for interp
    order = np.argsort(t_src)
    t = np.asarray(t_src)[order]
    y = np.asarray(y_src)[order]
    # Remove duplicates
    mask = np.diff(t, prepend=t[0] - 1e-12) > 0
    return np.interp(t_ref, t[mask], y[mask])


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    x = np.asarray(x)
    y = np.asarray(y)
    # Guard against zero variance
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def compute_for_traj(traj_dir: str, cfg_path: str, seed: int = 12345, crop_start_s: float = 0.0) -> Tuple[str, Dict[str, float]]:
    sim_path, real_path = find_log_pair(traj_dir)  # exists per caller
    clean_path = find_clean(traj_dir)
    if clean_path is None:
        raise FileNotFoundError(f"No clean_*.csv in {traj_dir}")

    # Load real
    _, real_df = load_and_prepare(sim_path, real_path)
    if crop_start_s > 0:
        real_df = real_df[real_df['t_norm'] >= crop_start_s].reset_index(drop=True)
    t_ref = real_df['t_norm'].to_numpy()

    # Synthesize and apply mapping
    synth_df = synth_from_clean(clean_path, cfg_path, seed=seed)
    if crop_start_s > 0:
        synth_df = synth_df[synth_df['t_norm'] >= crop_start_s].reset_index(drop=True)
    synth_df_al = apply_fixed_axis_map(synth_df)

    # Align series
    t_src = synth_df_al['time'].to_numpy() - synth_df_al['time'].iloc[0]

    pairs = [
        (('imu_ax',), ('acc_x_g',), 'ax'),
        (('imu_ay',), ('acc_y_g',), 'ay'),
        (('imu_az',), ('acc_z_g',), 'az'),
        (('imu_gx',), ('gyro_x_rad_s',), 'gx'),
        (('imu_gy',), ('gyro_y_rad_s',), 'gy'),
        (('imu_gz',), ('gyro_z_rad_s',), 'gz'),
    ]

    corr: Dict[str, float] = {}
    for (scol,), (rcol,), name in pairs:
        y_src = synth_df_al[scol].to_numpy()
        y_al = align_to(t_ref, t_src, y_src)
        r = real_df[rcol].to_numpy()
        corr[f'corr_{name}'] = corrcoef(y_al, r)

    return os.path.basename(traj_dir), corr


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Compute correlations (sim2real vs real) for logs/traj_*')
    ap.add_argument('--logs_root', default='logs')
    ap.add_argument('--config', default='imu_sim2real_plus/config/example_config.yaml')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default='runs/log_plots/correlations.csv')
    ap.add_argument('--crop_start_s', type=float, default=0.0)
    args = ap.parse_args()

    traj_dirs = sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d))
    rows: List[List[str]] = []
    header = ['traj', 'corr_ax', 'corr_ay', 'corr_az', 'corr_gx', 'corr_gy', 'corr_gz']
    rows.append(header)

    for d in traj_dirs:
        pair = find_log_pair(d)
        if not pair or find_clean(d) is None:
            continue
        name, vals = compute_for_traj(d, args.config, seed=args.seed, crop_start_s=args.crop_start_s)
        rows.append([
            name,
            *(f"{vals.get(k, float('nan')):.6f}" for k in header[1:])
        ])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        for row in rows:
            f.write(','.join(row) + '\n')
    print(f"Wrote correlations to {args.out}")

    # Also pretty-print
    for row in rows[1:]:
        print(row[0], {h: float(row[i]) for i, h in enumerate(header[1:], start=1)})


if __name__ == '__main__':
    main()
