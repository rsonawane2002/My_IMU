import argparse
import csv
from dataclasses import dataclass
import numpy as np


def read_csv(path: str):
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:]], dtype=float)
    return header, data


def write_csv(path: str, header, data: np.ndarray):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow([f"{x:.9f}" for x in row])


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def slerp(q0, q1, u: float):
    # q = w, x, y, z (scalar-first)
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = np.dot(q0, q1)
    # If dot < 0, take the shorter path by flipping q1
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Nearly identical: linear interpolate then renormalize
        q = q0 + u * (q1 - q0)
        return normalize_quat(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * u
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def resample_linear(t, x, t_new):
    return np.interp(t_new, t, x)


def resample_quat(t, q, t_new):
    # q: Nx4 array (w,x,y,z)
    q_new = np.empty((len(t_new), 4), dtype=float)
    idx = np.searchsorted(t, t_new, side='right')
    idx = np.clip(idx, 1, len(t) - 1)
    t0 = t[idx - 1]
    t1 = t[idx]
    u = (t_new - t0) / np.maximum(t1 - t0, 1e-12)
    for i in range(len(t_new)):
        q_new[i] = slerp(q[idx[i]-1], q[idx[i]], float(u[i]))
    return q_new


def main():
    ap = argparse.ArgumentParser(description='Downsample trajectory.csv to a target sampling rate.')
    ap.add_argument('--in', dest='inp', required=True, help='Input CSV (time, acc, gyro, quat)')
    ap.add_argument('--out', required=True, help='Output CSV path')
    ap.add_argument('--target_hz', type=float, required=True, help='Target sampling rate (Hz)')
    args = ap.parse_args()

    header, data = read_csv(args.inp)
    # Identify columns by header (input may contain many more; we extract the required fields)
    h = header
    time = data[:, h.index('time')]
    acc = data[:, [h.index('acc_x'), h.index('acc_y'), h.index('acc_z')]]
    gyro = data[:, [h.index('ang_vel_x'), h.index('ang_vel_y'), h.index('ang_vel_z')]]
    quat = data[:, [h.index('quat_w'), h.index('quat_x'), h.index('quat_y'), h.index('quat_z')]]

    dt_new = 1.0 / float(args.target_hz)
    t0, t1 = time[0], time[-1]
    t_new = np.arange(t0, t1 + 1e-12, dt_new)

    acc_new = np.column_stack([resample_linear(time, acc[:, i], t_new) for i in range(3)])
    gyro_new = np.column_stack([resample_linear(time, gyro[:, i], t_new) for i in range(3)])
    quat_new = resample_quat(time, quat, t_new)

    out = np.column_stack([t_new, acc_new, gyro_new, quat_new])
    header_out = ['time', 'acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
    write_csv(args.out, header_out, out)
    print(f"Wrote downsampled CSV: {args.out} (N={len(t_new)}, fs={args.target_hz} Hz)")


if __name__ == '__main__':
    main()
