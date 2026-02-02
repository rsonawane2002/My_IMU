import os
import glob
import numpy as np
import pandas as pd

from plot_all_logs_imu import (
    find_log_pair,
    load_and_prepare,
    compute_axis_alignment_separate,
    apply_axis_alignment_separate,
)


def corr(a, b):
    a = a - np.mean(a); b = b - np.mean(b)
    sa = np.std(a); sb = np.std(b)
    if sa == 0 or sb == 0:
        return float('nan')
    return float(np.corrcoef(a/sa, b/sb)[0, 1])


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Diagnose potential AY/AZ flips by comparing cross-correlations.')
    ap.add_argument('--logs_root', default='logs')
    ap.add_argument('--crop_start_s', type=float, default=0.4)
    ap.add_argument('--out', default='runs/axis_diagnostics.csv')
    args = ap.parse_args()

    rows = [['traj','corr(ay~AY)','corr(ay~AZ)','corr(az~AZ)','corr(az~AY)']]
    for traj in sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d)):
        pair = find_log_pair(traj)
        if not pair:
            continue
        sim_path, real_path = pair
        sim_df, real_df = load_and_prepare(sim_path, real_path)
        if args.crop_start_s > 0:
            sim_df = sim_df[sim_df['t_norm'] >= args.crop_start_s].reset_index(drop=True)
            real_df = real_df[real_df['t_norm'] >= args.crop_start_s].reset_index(drop=True)

        A_acc, A_gyr = compute_axis_alignment_separate(sim_df, real_df)
        sim_al = apply_axis_alignment_separate(sim_df, A_acc, A_gyr)

        t_r = real_df['t_norm'].to_numpy()
        def interp(col):
            return np.interp(t_r, sim_al['t_norm'].to_numpy(), sim_al[col].to_numpy())
        ay_sim = interp('imu_ay'); az_sim = interp('imu_az')
        ay_real = real_df['acc_y_g'].to_numpy(); az_real = real_df['acc_z_g'].to_numpy()

        rows.append([
            os.path.basename(traj),
            f"{corr(ay_sim, ay_real):.6f}",
            f"{corr(ay_sim, az_real):.6f}",
            f"{corr(az_sim, az_real):.6f}",
            f"{corr(az_sim, ay_real):.6f}",
        ])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(','.join(r) + '\n')
    print('Wrote', args.out)


if __name__ == '__main__':
    main()

