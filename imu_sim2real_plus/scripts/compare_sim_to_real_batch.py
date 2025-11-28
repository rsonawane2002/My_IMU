import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# Reuse helpers from root plotting script
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from plot_sim_vs_real import (
    ImuSeries,
    load_real_csv,
    load_sim_csv,
    interpolate_to,
    summarize_metrics,
    orthogonal_procrustes_with_bias,
    apply_rb,
    plot_overlays,
    plot_residuals,
)


def list_real_logs(root: Path) -> List[Path]:
    pats = list(root.rglob("real_imu_*.csv"))
    return sorted(pats)


def find_time_lag_seconds(real: ImuSeries, sim: ImuSeries, max_lag: float = 0.5) -> float:
    # Use gyro norm correlation to estimate lag; detrend both
    t_r = real.t
    t_s = sim.t
    # Create evenly spaced time base over overlap
    t0 = max(t_r.min(), t_s.min())
    t1 = min(t_r.max(), t_s.max())
    if t1 - t0 < 1e-3:
        return 0.0
    fs_guess = min(
        1.0 / np.median(np.diff(t_r)[1:]),
        1.0 / np.median(np.diff(t_s)[1:]),
    )
    fs = min(100.0, max(20.0, fs_guess))
    tt = np.arange(t0, t1, 1.0 / fs)
    def resample(series: ImuSeries):
        si = interpolate_to(tt, series)
        g = np.linalg.norm(np.stack([si.wx, si.wy, si.wz], axis=1), axis=1)
        g = g - g.mean()
        return g
    gr = resample(real)
    gs = resample(sim)
    # search lags
    max_shift = int(max_lag * fs)
    best_lag = 0
    best_corr = -np.inf
    for k in range(-max_shift, max_shift + 1):
        if k < 0:
            a = gr[-k:]
            b = gs[: len(a)]
        elif k > 0:
            a = gr[: len(gs) - k]
            b = gs[k:]
        else:
            a = gr
            b = gs
        if len(a) < fs * 0.5:
            continue
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        corr = float(a.dot(b) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = k
    return best_lag / fs


def _estimate_lag_single(real_sig: np.ndarray, sim_sig: np.ndarray, fs: float, max_lag: float) -> float:
    # Normalize
    a = real_sig - np.mean(real_sig)
    b = sim_sig - np.mean(sim_sig)
    if np.std(a) > 0:
        a = a / np.std(a)
    if np.std(b) > 0:
        b = b / np.std(b)
    max_shift = int(max_lag * fs)
    best_k = 0
    best_corr = -np.inf
    for k in range(-max_shift, max_shift + 1):
        if k < 0:
            aa = a[-k:]
            bb = b[: len(aa)]
        elif k > 0:
            aa = a[: len(b) - k]
            bb = b[k:]
        else:
            aa = a
            bb = b
        if len(aa) < fs * 0.5:
            continue
        corr = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_k = k
    # Parabolic refine around best_k for sub-sample precision when possible
    def corr_at(k):
        if k < 0:
            aa = a[-k:]
            bb = b[: len(aa)]
        elif k > 0:
            aa = a[: len(b) - k]
            bb = b[k:]
        else:
            aa = a
            bb = b
        return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
    c0 = corr_at(best_k)
    cm = corr_at(best_k - 1)
    cp = corr_at(best_k + 1)
    denom = (cm - 2 * c0 + cp)
    frac = 0.0
    if abs(denom) > 1e-12:
        frac = 0.5 * (cm - cp) / denom
        frac = np.clip(frac, -1.0, 1.0)
    return (best_k + frac) / fs


def estimate_lags_per_axis(real: ImuSeries, sim: ImuSeries, max_lag: float = 2.5) -> dict:
    # Build common uniform time grid over overlap
    t0 = max(real.t.min(), sim.t.min())
    t1 = min(real.t.max(), sim.t.max())
    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 - t0) <= 0:
        return {}
    fs_guess = min(
        1.0 / np.median(np.diff(real.t)[1:]),
        1.0 / np.median(np.diff(sim.t)[1:]),
    )
    fs = min(200.0, max(40.0, fs_guess))
    tt = np.arange(t0, t1, 1.0 / fs)
    r = interpolate_to(tt, real)
    s = interpolate_to(tt, sim)
    lags = {}
    # Gyro axes (primary for lag)
    lags['gyro_x_s'] = _estimate_lag_single(r.wx, s.wx, fs, max_lag)
    lags['gyro_y_s'] = _estimate_lag_single(r.wy, s.wy, fs, max_lag)
    lags['gyro_z_s'] = _estimate_lag_single(r.wz, s.wz, fs, max_lag)
    # Accel axes (optional diagnostic)
    lags['acc_x_s'] = _estimate_lag_single(r.ax, s.ax, fs, max_lag)
    lags['acc_y_s'] = _estimate_lag_single(r.ay, s.ay, fs, max_lag)
    lags['acc_z_s'] = _estimate_lag_single(r.az, s.az, fs, max_lag)
    # Aggregate suggestion (median of gyro)
    lags['suggested_lag_s'] = float(np.median([lags['gyro_x_s'], lags['gyro_y_s'], lags['gyro_z_s']]))
    lags['fs_used_hz'] = fs
    return lags


def load_clean_csv(path: str) -> ImuSeries:
    # Headers: time, ..., acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        t: List[float] = []
        ax: List[float] = []
        ay: List[float] = []
        az: List[float] = []
        wx: List[float] = []
        wy: List[float] = []
        wz: List[float] = []
        for row in reader:
            try:
                t.append(float(row.get("time", row.get("time_sec", row.get("t", "nan")))) )
                ax.append(float(row["acc_x"]))
                ay.append(float(row["acc_y"]))
                az.append(float(row["acc_z"]))
                wx.append(float(row["ang_vel_x"]))
                wy.append(float(row["ang_vel_y"]))
                wz.append(float(row["ang_vel_z"]))
            except Exception:
                continue
    return ImuSeries(
        t=np.asarray(t, dtype=float),
        ax=np.asarray(ax, dtype=float),
        ay=np.asarray(ay, dtype=float),
        az=np.asarray(az, dtype=float),
        wx=np.asarray(wx, dtype=float),
        wy=np.asarray(wy, dtype=float),
        wz=np.asarray(wz, dtype=float),
    )


def map_clean_from_real(real_csv: Path) -> Path:
    # Replace real_imu_N.csv -> clean_N.csv inside same folder
    name = real_csv.name
    if name.startswith("real_imu_") and name.endswith(".csv"):
        idx = name[len("real_imu_") : -len(".csv")]
        cand = real_csv.parent / f"clean_{idx}.csv"
        if cand.exists():
            return cand
    # Fallback: first clean_*.csv in the folder
    cands = sorted(real_csv.parent.glob("clean_*.csv"))
    return cands[0] if cands else real_csv


def estimate_lag_from_clean(real: ImuSeries, clean: ImuSeries, max_lag: float = 2.5) -> Tuple[float, str, float]:
    # Uniform grid over overlap
    t0 = max(real.t.min(), clean.t.min())
    t1 = min(real.t.max(), clean.t.max())
    if (t1 - t0) <= 0:
        return 0.0, "none", 0.0
    fs_guess = min(
        1.0 / np.median(np.diff(real.t)[1:]),
        1.0 / np.median(np.diff(clean.t)[1:]),
    )
    fs = min(200.0, max(40.0, fs_guess))
    tt = np.arange(t0, t1, 1.0 / fs)
    r = interpolate_to(tt, real)
    c = interpolate_to(tt, clean)
    # Pick gyro axis with highest variance in clean
    axes = {
        "gyro_x": c.wx,
        "gyro_y": c.wy,
        "gyro_z": c.wz,
    }
    variances = {k: float(np.var(v)) for k, v in axes.items()}
    best_axis = max(variances.items(), key=lambda kv: kv[1])[0]
    r_sig = getattr(r, {"gyro_x": "wx", "gyro_y": "wy", "gyro_z": "wz"}[best_axis])
    c_sig = axes[best_axis]
    lag = _estimate_lag_single(r_sig, c_sig, fs, max_lag)
    return lag, best_axis, variances[best_axis]


def run_one(real_csv: Path, sim_csv: Path, out_dir: Path, estimate_lag: bool = False, max_lag: float = 2.5, use_clean_for_lag: bool = True) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    real = load_real_csv(str(real_csv))
    sim = load_sim_csv(str(sim_csv))

    # Derive lag either from clean-vs-real or simple sim-vs-real matching
    lag = 0.0
    lag_source = "none"
    best_axis = ""
    best_axis_var = 0.0
    sim_shifted = sim
    if use_clean_for_lag:
        clean_csv = map_clean_from_real(real_csv)
        if clean_csv.exists():
            clean = load_clean_csv(str(clean_csv))
            lag, best_axis, best_axis_var = estimate_lag_from_clean(real, clean, max_lag=max_lag)
            lag_source = f"clean_vs_real({best_axis})"
            sim_shifted = ImuSeries(
                t=sim.t + lag,
                ax=sim.ax, ay=sim.ay, az=sim.az,
                wx=sim.wx, wy=sim.wy, wz=sim.wz,
            )
    elif estimate_lag:
        lag = find_time_lag_seconds(real, sim, max_lag=min(0.5, max_lag))
        lag_source = "sim_vs_real_norm"
        sim_shifted = ImuSeries(
            t=sim.t + lag,
            ax=sim.ax, ay=sim.ay, az=sim.az,
            wx=sim.wx, wy=sim.wy, wz=sim.wz,
        )

    sim_i = interpolate_to(real.t, sim_shifted)

    # Metrics before transform
    m0 = summarize_metrics(real, sim_i)

    # Fit rotation + bias (per-sequence)
    sim_acc = np.stack([sim_i.ax, sim_i.ay, sim_i.az], axis=1)
    real_acc = np.stack([real.ax, real.ay, real.az], axis=1)
    sim_gyr = np.stack([sim_i.wx, sim_i.wy, sim_i.wz], axis=1)
    real_gyr = np.stack([real.wx, real.wy, real.wz], axis=1)
    R_a, b_a = orthogonal_procrustes_with_bias(sim_acc, real_acc)
    R_w, b_w = orthogonal_procrustes_with_bias(sim_gyr, real_gyr)
    sim_tf = ImuSeries(
        t=real.t,
        ax=apply_rb(R_a, b_a, sim_acc)[:, 0],
        ay=apply_rb(R_a, b_a, sim_acc)[:, 1],
        az=apply_rb(R_a, b_a, sim_acc)[:, 2],
        wx=apply_rb(R_w, b_w, sim_gyr)[:, 0],
        wy=apply_rb(R_w, b_w, sim_gyr)[:, 1],
        wz=apply_rb(R_w, b_w, sim_gyr)[:, 2],
    )
    m1 = summarize_metrics(real, sim_tf)

    # Estimate per-axis lags after transformation
    lags = estimate_lags_per_axis(real, sim_tf, max_lag=max_lag)
    lag_med = lags.get('suggested_lag_s', 0.0)
    # Apply lag correction and recompute metrics
    sim_tf_lag = ImuSeries(
        t=sim_tf.t + lag_med,
        ax=sim_tf.ax, ay=sim_tf.ay, az=sim_tf.az,
        wx=sim_tf.wx, wy=sim_tf.wy, wz=sim_tf.wz,
    )
    sim_tf_lag_i = interpolate_to(real.t, sim_tf_lag)
    m2 = summarize_metrics(real, sim_tf_lag_i)

    # Save metrics and plots
    with open(out_dir / "metrics_before.txt", "w") as f:
        for k in sorted(m0.keys()):
            f.write(f"{k}: {m0[k]:.6f}\n")
        f.write(f"time_lag_seconds_applied: {lag:.6f}\n")
        f.write(f"lag_source: {lag_source}\n")
        if best_axis:
            f.write(f"lag_axis: {best_axis}\nlag_axis_variance: {best_axis_var:.6f}\n")
    with open(out_dir / "metrics_after_transform.txt", "w") as f:
        for k in sorted(m1.keys()):
            f.write(f"{k}: {m1[k]:.6f}\n")
        f.write(f"time_lag_seconds: {lag:.6f}\n")
    # Save lag diagnostics and metrics after lag correction
    with open(out_dir / "lags_seconds.txt", "w") as f:
        for k, v in lags.items():
            f.write(f"{k}: {v:.6f}\n" if isinstance(v, (float, int)) else f"{k}: {v}\n")
        f.write(f"lag_from_clean_seconds: {lag:.6f}\n")
        f.write(f"lag_from_clean_source: {lag_source}\n")
        if best_axis:
            f.write(f"lag_from_clean_axis: {best_axis}\nlag_from_clean_axis_variance: {best_axis_var:.6f}\n")
    with open(out_dir / "metrics_after_transform_lag.txt", "w") as f:
        for k in sorted(m2.keys()):
            f.write(f"{k}: {m2[k]:.6f}\n")
        f.write(f"applied_lag_seconds: {lag_med:.6f}\n")
    np.savetxt(out_dir / "fit_acc_R.txt", R_a)
    np.savetxt(out_dir / "fit_acc_b.txt", b_a[None, :])
    np.savetxt(out_dir / "fit_gyr_R.txt", R_w)
    np.savetxt(out_dir / "fit_gyr_b.txt", b_w[None, :])

    plot_overlays(real, sim_i, str(out_dir / "overlay_before.png"))
    plot_residuals(real, sim_i, str(out_dir / "residuals_before.png"))
    plot_overlays(real, sim_tf, str(out_dir / "overlay_after.png"))
    plot_residuals(real, sim_tf, str(out_dir / "residuals_after.png"))
    plot_overlays(real, sim_tf_lag_i, str(out_dir / "overlay_after_lagcorrected.png"))
    plot_residuals(real, sim_tf_lag_i, str(out_dir / "residuals_after_lagcorrected.png"))

    # Return aggregate for CSV
    agg = {**{f"before_{k}": v for k, v in m0.items()}, **{f"after_{k}": v for k, v in m1.items()}, **{f"after_lag_{k}": v for k, v in m2.items()}}
    agg["time_lag_seconds_initial"] = lag
    agg["time_lag_seconds_per_axis_median"] = lag_med
    agg["lag_source"] = lag_source
    if best_axis:
        agg["lag_axis"] = best_axis
        agg["lag_axis_var"] = best_axis_var
    return agg


def main():
    ap = argparse.ArgumentParser(description="Batch compare sim vs real logs.")
    ap.add_argument("--logs_root", default="logs", help="Root folder containing real_imu_*.csv files")
    ap.add_argument("--sim_csv", default="runs/log_noise_clean1/seq_00000_model_output.csv")
    ap.add_argument("--out_root", default="runs/real_compare_batch")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--estimate_lag", action="store_true")
    ap.add_argument("--max_lag_seconds", type=float, default=2.5)
    ap.add_argument("--no_clean_lag", action="store_true", help="Disable using clean_* to compute lag; fall back to sim-vs-real")
    args = ap.parse_args()

    logs = list_real_logs(Path(args.logs_root))
    if not logs:
        print("No real_imu_*.csv files found under", args.logs_root)
        return
    random.seed(args.seed)
    picks = random.sample(logs, k=min(args.num, len(logs)))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float]] = []
    for i, p in enumerate(picks):
        tag = p.parent.name + "_" + p.stem
        out_dir = out_root / f"{i:02d}_{tag}"
        print(f"Processing {p} -> {out_dir}")
        agg = run_one(p, Path(args.sim_csv), out_dir, estimate_lag=args.estimate_lag, max_lag=args.max_lag_seconds, use_clean_for_lag=not args.no_clean_lag)
        agg_row = {"log": str(p)}
        agg_row.update(agg)
        summary_rows.append(agg_row)

    # Write summary CSV
    keys = sorted({k for row in summary_rows for k in row.keys()})
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print("Wrote", out_root / "summary.csv")


if __name__ == "__main__":
    main()
