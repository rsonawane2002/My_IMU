import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np
import yaml
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


def find_clean(traj_dir: str) -> Optional[str]:
    """Return path to clean_*.csv in a traj directory if present."""
    cands = sorted(glob.glob(os.path.join(traj_dir, 'clean_*.csv')))
    return cands[0] if cands else None


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

    # Real accelerometer columns are already in m/s^2 per dataset notes

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

    # Accelerometer (m/s^2)
    for i, (col_sim, col_real, label) in enumerate(zip(SIM_COLS[1:4], REAL_COLS[1:4], ['ax', 'ay', 'az'])):
        if col_sim in sim_df and col_real in real_df:
            axes[i].plot(sim_df['t_norm'], sim_df[col_sim], label='Sim ' + label)
            axes[i].plot(real_df['t_norm'], real_df[col_real], label='Real ' + label)
        axes[i].set_title(f'IMU {label.upper()}')
        axes[i].set_xlabel('Time [s]')
        axes[i].set_ylabel('Acceleration [m/s^2]')
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


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def synth_from_clean(clean_csv: str, cfg_path: str, seed: int = 12345) -> pd.DataFrame:
    """Load clean_*.csv and synthesize noisy IMU via sim2real pipeline.

    Returns DataFrame with columns matching SIM_COLS names.
    """
    from imu_sim2real_plus.sensors.imu_synth import synth_measurements

    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Ensure numeric noise_density
    cfg['imu']['gyro']['noise_density'][0] = float(cfg['imu']['gyro']['noise_density'][0])
    cfg['imu']['gyro']['noise_density'][1] = float(cfg['imu']['gyro']['noise_density'][1])
    cfg['imu']['accel']['noise_density'][0] = float(cfg['imu']['accel']['noise_density'][0])
    cfg['imu']['accel']['noise_density'][1] = float(cfg['imu']['accel']['noise_density'][1])

    df = pd.read_csv(clean_csv)
    df.columns = df.columns.str.strip()

    req = ['time','acc_x','acc_y','acc_z','ang_vel_x','ang_vel_y','ang_vel_z','quat_w','quat_x','quat_y','quat_z']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(clean_csv)} missing columns: {missing}")

    t = df['time'].to_numpy()
    dt = float(np.mean(np.diff(t)))

    a_B_input = df[['acc_x','acc_y','acc_z']].to_numpy()
    w_B = df[['ang_vel_x','ang_vel_y','ang_vel_z']].to_numpy()

    quats = df[['quat_w','quat_x','quat_y','quat_z']].to_numpy()
    R_WB = np.array([quaternion_to_rotation_matrix(q) for q in quats])
    R_BW = np.transpose(R_WB, (0,2,1))

    # Derive a_W so that specific force from the synthesizer matches the body-frame clean input:
    # We want R_BW (a_W - g_W) ~= a_B - g_B  => a_W = R_WB (a_B - g_B) + g_W
    g_W = np.array([0.0, 0.0, 9.81])
    g_B = np.einsum('nij,j->ni', R_WB, g_W)
    f_B_true = a_B_input - g_B
    a_W = np.einsum('nij,ni->nj', R_WB, f_B_true) + g_W
    wdot_B = np.gradient(w_B, dt, axis=0)

    r_lever = np.array([0.0, 0.0, 0.01])
    seq, _params = synth_measurements(R_WB, w_B, a_W, wdot_B, r_lever, cfg, dt, seed=seed)

    # Convert to plotting DataFrame (accel in m/s^2, gyro in rad/s)
    f_meas = seq['f_meas']
    w_meas = seq['w_meas']

    out = pd.DataFrame({
        'time': t,
        'imu_ax': f_meas[:,0],
        'imu_ay': f_meas[:,1],
        'imu_az': f_meas[:,2],
        'imu_gx': w_meas[:,0],
        'imu_gy': w_meas[:,1],
        'imu_gz': w_meas[:,2],
    })
    out['t_norm'] = out['time'] - out['time'].iloc[0]
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x - np.mean(x); y = y - np.mean(y)
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x/sx, y/sy)[0,1])


def _interp_to(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(t_dst, t_src, v_src[:,i], left=v_src[0,i], right=v_src[-1,i]) for i in range(v_src.shape[1])]).T


def compute_axis_alignment(sim_df: pd.DataFrame, real_df: pd.DataFrame) -> np.ndarray:
    """Compute best signed permutation matrix A (3x3) so that
    sim @ A aligns to real across both accel and gyro using correlation.
    """
    from itertools import permutations

    t_sim = sim_df['t_norm'].to_numpy()
    t_real = real_df['t_norm'].to_numpy()
    X_acc = sim_df[['imu_ax','imu_ay','imu_az']].to_numpy()
    X_gyr = sim_df[['imu_gx','imu_gy','imu_gz']].to_numpy()
    Y_acc = real_df[['acc_x_g','acc_y_g','acc_z_g']].to_numpy()
    Y_gyr = real_df[['gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s']].to_numpy()

    # Interpolate sim onto real timeline
    X_acc_i = _interp_to(t_sim, X_acc, t_real)
    X_gyr_i = _interp_to(t_sim, X_gyr, t_real)

    # Correlation matrix between each axis
    def corr_mat(X, Y):
        C = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                C[i,j] = _safe_corr(X[:,j], Y[:,i])
        return C

    C_acc = corr_mat(X_acc_i, Y_acc)
    C_gyr = corr_mat(X_gyr_i, Y_gyr)
    C = C_acc + C_gyr  # combine evidence

    best_score = -np.inf
    best_perm = None
    best_signs = None
    for p in permutations(range(3)):
        signs = []
        score = 0.0
        for i in range(3):
            c = C[i, p[i]]
            s = 1.0 if c >= 0 else -1.0
            signs.append(s)
            score += abs(c)
        if score > best_score:
            best_score = score
            best_perm = p
            best_signs = signs
    A = np.zeros((3,3))
    for i in range(3):
        j = best_perm[i]
        A[j, i] = best_signs[i]
    return A


def apply_axis_alignment(df: pd.DataFrame, A: np.ndarray) -> pd.DataFrame:
    acc = df[['imu_ax','imu_ay','imu_az']].to_numpy()
    gyr = df[['imu_gx','imu_gy','imu_gz']].to_numpy()
    acc_al = acc @ A
    gyr_al = gyr @ A
    out = df.copy()
    out['imu_ax'], out['imu_ay'], out['imu_az'] = acc_al[:,0], acc_al[:,1], acc_al[:,2]
    out['imu_gx'], out['imu_gy'], out['imu_gz'] = gyr_al[:,0], gyr_al[:,1], gyr_al[:,2]
    return out


def apply_fixed_axis_map(df: pd.DataFrame) -> pd.DataFrame:
    """Apply user-specified fixed axis transform to simulated data.

    Mapping provided:
      Accelerometer (keep from prior step):
        AX_new = -AZ_old
        AY_new = -AY_old
        AZ_new =  AX_old
      Gyroscope:
        GX_new = -GZ_old
        GY_new = -GY_old
        GZ_new = -GX_old
    """
    out = df.copy()
    ax = df['imu_ax'].to_numpy()
    ay = df['imu_ay'].to_numpy()
    az = df['imu_az'].to_numpy()
    gx = df['imu_gx'].to_numpy()
    gy = df['imu_gy'].to_numpy()
    gz = df['imu_gz'].to_numpy()

    out['imu_ax'] = -az
    out['imu_ay'] = -ay
    out['imu_az'] = ax
    out['imu_gx'] = -gz
    out['imu_gy'] = -gy
    out['imu_gz'] = -gx
    return out


def main():
    parser = argparse.ArgumentParser(description='Plot IMU sim vs real for all logs/traj_* directories.')
    parser.add_argument('--logs_root', default='logs', help='Root containing traj_* subdirectories')
    parser.add_argument('--out', default='runs/log_plots', help='Output directory for plots')
    parser.add_argument('--config', default='imu_sim2real_plus/config/example_config.yaml', help='IMU noise config for sim2real synthetic variant')
    parser.add_argument('--seed', type=int, default=12345, help='RNG seed for synthetic noise')
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
        base = os.path.basename(traj)

        # 1) Plot original sim imu_*.csv vs real
        try:
            sim_df, real_df = load_and_prepare(sim_path, real_path)
            # Apply fixed axis mapping to the simulated data
            sim_df_al = apply_fixed_axis_map(sim_df)
            out_path = os.path.join(args.out, f'{base}.png')
            title = f'{base}: {os.path.basename(sim_path)} vs {os.path.basename(real_path)} (fixed-axis)'
            plot_pair(sim_df_al, real_df, title, out_path)
            print(f'Wrote {out_path}')
        except Exception as e:
            print(f'Failed to plot {traj} (sim vs real): {e}')

        # 2) If clean exists, run sim2real pipeline and plot against real
        clean_path = find_clean(traj)
        if clean_path:
            try:
                synth_df = synth_from_clean(clean_path, args.config, seed=args.seed)
                synth_df_al = apply_fixed_axis_map(synth_df)
                # real_df already loaded above; ensure exists
                if 'real_df' not in locals():
                    _, real_df = load_and_prepare(sim_path, real_path)
                out_path2 = os.path.join(args.out, f'{base}_sim2real.png')
                title2 = f'{base}: sim2real({os.path.basename(clean_path)}) vs {os.path.basename(real_path)} (fixed-axis)'
                plot_pair(synth_df_al, real_df, title2, out_path2)
                print(f'Wrote {out_path2}')
            except Exception as e:
                print(f'Failed sim2real on {traj}: {e}')


if __name__ == '__main__':
    main()
