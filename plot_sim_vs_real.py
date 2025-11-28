import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ImuSeries:
    t: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    wx: np.ndarray
    wy: np.ndarray
    wz: np.ndarray


def load_real_csv(path: str) -> ImuSeries:
    # Expected headers: time_sec,acc_x_g,acc_y_g,acc_z_g,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s
    cols = [
        "time_sec",
        "acc_x_g",
        "acc_y_g",
        "acc_z_g",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
    ]
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        data: Dict[str, List[float]] = {c: [] for c in cols}
        for row in reader:
            try:
                data["time_sec"].append(float(row.get("time_sec", row.get("time", row.get("t", "nan")))))
                data["acc_x_g"].append(float(row["acc_x_g"]))
                data["acc_y_g"].append(float(row["acc_y_g"]))
                data["acc_z_g"].append(float(row["acc_z_g"]))
                data["gyro_x_rad_s"].append(float(row["gyro_x_rad_s"]))
                data["gyro_y_rad_s"].append(float(row["gyro_y_rad_s"]))
                data["gyro_z_rad_s"].append(float(row["gyro_z_rad_s"]))
            except Exception:
                # skip malformed rows
                continue
    t = np.asarray(data["time_sec"], dtype=float)
    return ImuSeries(
        t=t,
        ax=np.asarray(data["acc_x_g"], dtype=float),
        ay=np.asarray(data["acc_y_g"], dtype=float),
        az=np.asarray(data["acc_z_g"], dtype=float),
        wx=np.asarray(data["gyro_x_rad_s"], dtype=float),
        wy=np.asarray(data["gyro_y_rad_s"], dtype=float),
        wz=np.asarray(data["gyro_z_rad_s"], dtype=float),
    )


def load_sim_csv(path: str) -> ImuSeries:
    # Expected headers: time,f_x,f_y,f_z,w_x,w_y,w_z
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        data: Dict[str, List[float]] = {k: [] for k in ["time", "f_x", "f_y", "f_z", "w_x", "w_y", "w_z"]}
        for row in reader:
            try:
                data["time"].append(float(row.get("time", row.get("time_sec", row.get("t", "nan")))))
                data["f_x"].append(float(row["f_x"]))
                data["f_y"].append(float(row["f_y"]))
                data["f_z"].append(float(row["f_z"]))
                data["w_x"].append(float(row["w_x"]))
                data["w_y"].append(float(row["w_y"]))
                data["w_z"].append(float(row["w_z"]))
            except Exception:
                continue
    t = np.asarray(data["time"], dtype=float)
    return ImuSeries(
        t=t,
        ax=np.asarray(data["f_x"], dtype=float),
        ay=np.asarray(data["f_y"], dtype=float),
        az=np.asarray(data["f_z"], dtype=float),
        wx=np.asarray(data["w_x"], dtype=float),
        wy=np.asarray(data["w_y"], dtype=float),
        wz=np.asarray(data["w_z"], dtype=float),
    )


def interpolate_to(t_ref: np.ndarray, series: ImuSeries) -> ImuSeries:
    # Guard monotonicity and uniqueness for interpolation
    order = np.argsort(series.t)
    t = series.t[order]
    # Remove duplicates
    mask = np.diff(t, prepend=t[0] - 1e-12) > 0
    t = t[mask]
    def interp(x):
        return np.interp(t_ref, t, x[order][mask])
    return ImuSeries(
        t=t_ref,
        ax=interp(series.ax),
        ay=interp(series.ay),
        az=interp(series.az),
        wx=interp(series.wx),
        wy=interp(series.wy),
        wz=interp(series.wz),
    )


def plot_overlays(real: ImuSeries, simi: ImuSeries, out_path: str):
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    pairs = [
        (real.ax, simi.ax, "Accel X [m/s^2]"),
        (real.wx, simi.wx, "Gyro X [rad/s]"),
        (real.ay, simi.ay, "Accel Y [m/s^2]"),
        (real.wy, simi.wy, "Gyro Y [rad/s]"),
        (real.az, simi.az, "Accel Z [m/s^2]"),
        (real.wz, simi.wz, "Gyro Z [rad/s]"),
    ]
    t = real.t
    for i, (r, s, title) in enumerate(pairs):
        ax = axes[i // 2, i % 2]
        ax.plot(t, r, label="Real", lw=1.0)
        ax.plot(t, s, label="Sim->Real t", lw=1.0, alpha=0.8)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if i // 2 == 2:
            ax.set_xlabel("Time [s]")
        if i == 0:
            ax.legend(loc="upper right")
    fig.suptitle("IMU: Real vs Sim (interpolated to real timestamps)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(real: ImuSeries, simi: ImuSeries, out_path: str):
    res = [
        (real.ax - simi.ax, "Accel X Residual [m/s^2]"),
        (real.ay - simi.ay, "Accel Y Residual [m/s^2]"),
        (real.az - simi.az, "Accel Z Residual [m/s^2]"),
        (real.wx - simi.wx, "Gyro X Residual [rad/s]"),
        (real.wy - simi.wy, "Gyro Y Residual [rad/s]"),
        (real.wz - simi.wz, "Gyro Z Residual [rad/s]"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    t = real.t
    for i, (r, title) in enumerate(res):
        ax = axes[i // 2, i % 2]
        ax.plot(t, r, lw=0.8)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if i // 2 == 2:
            ax.set_xlabel("Time [s]")
    fig.suptitle("Residuals: Real - Sim (aligned)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_metrics(real: ImuSeries, simi: ImuSeries) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, r, s in [
        ("ax", real.ax, simi.ax),
        ("ay", real.ay, simi.ay),
        ("az", real.az, simi.az),
        ("wx", real.wx, simi.wx),
        ("wy", real.wy, simi.wy),
        ("wz", real.wz, simi.wz),
    ]:
        d = r - s
        out[f"{name}_mae"] = float(np.mean(np.abs(d)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(d ** 2)))
        out[f"{name}_bias"] = float(np.mean(d))
    return out


def orthogonal_procrustes_with_bias(sim_xyz: np.ndarray, real_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for R (SO3) and bias b so that R @ sim + b ≈ real in LS sense."""
    # Shapes: (N,3)
    X = np.asarray(sim_xyz)
    Y = np.asarray(real_xyz)
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    # Covariance for Procrustes
    C = Yc.T @ Xc  # (3,3)
    U, S, Vt = np.linalg.svd(C)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    b = mu_y - R @ mu_x
    return R, b


def apply_rb(R: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (R @ x.T).T + b


def main():
    ap = argparse.ArgumentParser(description="Compare sim IMU vs real IMU and plot gaps.")
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--sim_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fit_transform", action="store_true", help="Estimate best-fit rotation+bias and replot transformed sim")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    real = load_real_csv(args.real_csv)
    sim = load_sim_csv(args.sim_csv)

    # Interpolate sim onto real timestamps for fair comparison
    sim_i = interpolate_to(real.t, sim)

    # Basic summary and save plots
    metrics = summarize_metrics(real, sim_i)
    with open(os.path.join(args.out_dir, "metrics_summary.txt"), "w") as f:
        for k in sorted(metrics.keys()):
            f.write(f"{k}: {metrics[k]:.6f}\n")

    plot_overlays(real, sim_i, os.path.join(args.out_dir, "overlay_real_vs_sim.png"))
    plot_residuals(real, sim_i, os.path.join(args.out_dir, "residuals_real_minus_sim.png"))

    if args.fit_transform:
        # Stack vectors
        sim_acc = np.stack([sim_i.ax, sim_i.ay, sim_i.az], axis=1)
        real_acc = np.stack([real.ax, real.ay, real.az], axis=1)
        sim_gyr = np.stack([sim_i.wx, sim_i.wy, sim_i.wz], axis=1)
        real_gyr = np.stack([real.wx, real.wy, real.wz], axis=1)

        R_a, b_a = orthogonal_procrustes_with_bias(sim_acc, real_acc)
        R_w, b_w = orthogonal_procrustes_with_bias(sim_gyr, real_gyr)

        sim_acc_tf = apply_rb(R_a, b_a, sim_acc)
        sim_gyr_tf = apply_rb(R_w, b_w, sim_gyr)

        sim_tf = ImuSeries(
            t=real.t,
            ax=sim_acc_tf[:, 0],
            ay=sim_acc_tf[:, 1],
            az=sim_acc_tf[:, 2],
            wx=sim_gyr_tf[:, 0],
            wy=sim_gyr_tf[:, 1],
            wz=sim_gyr_tf[:, 2],
        )

        metr2 = summarize_metrics(real, sim_tf)
        with open(os.path.join(args.out_dir, "metrics_summary_transformed.txt"), "w") as f:
            for k in sorted(metr2.keys()):
                f.write(f"{k}: {metr2[k]:.6f}\n")
        np.savetxt(os.path.join(args.out_dir, "fit_acc_R.txt"), R_a)
        np.savetxt(os.path.join(args.out_dir, "fit_acc_b.txt"), b_a[None, :])
        np.savetxt(os.path.join(args.out_dir, "fit_gyr_R.txt"), R_w)
        np.savetxt(os.path.join(args.out_dir, "fit_gyr_b.txt"), b_w[None, :])

        plot_overlays(real, sim_tf, os.path.join(args.out_dir, "overlay_real_vs_sim_transformed.png"))
        plot_residuals(real, sim_tf, os.path.join(args.out_dir, "residuals_real_minus_sim_transformed.png"))

    # Also store aligned CSV for debugging
    aligned_csv = os.path.join(args.out_dir, "aligned_series.csv")
    with open(aligned_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time",
            "real_ax",
            "real_ay",
            "real_az",
            "real_wx",
            "real_wy",
            "real_wz",
            "sim_ax",
            "sim_ay",
            "sim_az",
            "sim_wx",
            "sim_wy",
            "sim_wz",
        ])
        for i in range(len(real.t)):
            w.writerow([
                f"{real.t[i]:.9f}",
                f"{real.ax[i]:.9f}",
                f"{real.ay[i]:.9f}",
                f"{real.az[i]:.9f}",
                f"{real.wx[i]:.9f}",
                f"{real.wy[i]:.9f}",
                f"{real.wz[i]:.9f}",
                f"{sim_i.ax[i]:.9f}",
                f"{sim_i.ay[i]:.9f}",
                f"{sim_i.az[i]:.9f}",
                f"{sim_i.wx[i]:.9f}",
                f"{sim_i.wy[i]:.9f}",
                f"{sim_i.wz[i]:.9f}",
            ])

    print("Wrote:", os.path.join(args.out_dir, "overlay_real_vs_sim.png"))
    print("Wrote:", os.path.join(args.out_dir, "residuals_real_minus_sim.png"))
    print("Wrote:", aligned_csv)


if __name__ == "__main__":
    main()
