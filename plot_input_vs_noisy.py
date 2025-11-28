import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_arrays(out_dir: str, seq: int):
    base = os.path.join(out_dir, f"seq_{seq:05d}")
    arr = lambda k: np.load(base + f"_{k}.npy")
    with open(base + "_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    data = {
        "f_meas": arr("f_meas"),
        "w_meas": arr("w_meas"),
        "gt_a_W": arr("gt_a_W"),
        "gt_w_B": arr("gt_w_B"),
        "gt_R_WB": arr("gt_R_WB"),
        "input_a_B": arr("input_a_B"),
    }
    return data, meta


def compute_input_specific_force(input_a_B: np.ndarray, R_WB: np.ndarray):
    # World is NED: g_W = +9.81 along +Z (Down)
    # Specific force: f_B = a_B - g_B, with g_B = R_WB @ g_W
    g_W = np.array([0.0, 0.0, 9.81])
    g_B = R_WB @ g_W
    return input_a_B - g_B


def main():
    p = argparse.ArgumentParser(description="Compare input vs. noisy IMU output with interactive plots.")
    p.add_argument("--dir", default="runs/from_log", help="Output directory containing seq_XXXXX files")
    p.add_argument("--seq", type=int, default=0, help="Sequence index (default: 0)")
    p.add_argument("--style", default="seaborn-v0_8-darkgrid", help="Matplotlib style (default: seaborn-v0_8-darkgrid)")
    p.add_argument("--residuals", action="store_true", help="Show residuals row (differences)")
    p.add_argument("--linky", action="store_true", help="Link Y limits across rows for same quantity")
    args = p.parse_args()

    plt.style.use(args.style)

    data, meta = load_arrays(args.dir, args.seq)
    f_meas = data["f_meas"]; w_meas = data["w_meas"]
    gt_w_B = data["gt_w_B"]; R_WB = data["gt_R_WB"]; input_a_B = data["input_a_B"]

    # Derive the input specific force that corresponds to the CSV input
    input_f_B = compute_input_specific_force(input_a_B, R_WB)

    # Time vector
    N = f_meas.shape[0]
    odr = meta.get("odr", None)
    if odr is None or odr <= 0:
        # Fallback from seconds if present
        seconds = meta.get("seconds", None)
        if seconds is not None and seconds > 0:
            odr = N / seconds
        else:
            odr = 100.0
    t = np.arange(N) / float(odr)

    # Metrics
    rms_acc = np.sqrt(np.mean((f_meas - input_f_B) ** 2, axis=0))
    rms_gyro = np.sqrt(np.mean((w_meas - gt_w_B) ** 2, axis=0))
    print("RMS specific force [m/s^2] per axis:", np.round(rms_acc, 5))
    print("RMS angular rate  [rad/s]  per axis:", np.round(rms_gyro, 5))

    # Plot: columns = axes (X,Y,Z), rows = accel specific force vs angular rate (+ optional residuals)
    nrows = 3 if args.residuals else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 8 if args.residuals else 6), sharex=True, sharey='row')
    fig.suptitle("Input vs. Realistic IMU Output", fontsize=14, weight="bold")

    accel_labels = ["X", "Y", "Z"]
    # Colorblind-friendly palette
    colors = {
        "input": "#0173b2",   # blue
        "noisy": "#de8f05",   # orange
        "noisy_gyro": "#d55e00",  # red-orange for gyro
        "residual": "#029e73", # green
    }

    # Specific force row
    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, input_f_B[:, i], color=colors["input"], lw=1.2, ls="--", label="input f_B")
        ax.plot(t, f_meas[:, i], color=colors["noisy"], lw=1.4, alpha=0.9, label="noisy f_B")
        ax.set_title(f"Specific Force — {accel_labels[i]}")
        ax.set_ylabel("m/s²")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.minorticks_on()

    # Angular rate row
    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, data["gt_w_B"][:, i], color=colors["input"], lw=1.2, ls="--", label="input w_B")
        ax.plot(t, w_meas[:, i], color=colors["noisy_gyro"], lw=1.4, alpha=0.9, label="noisy w_B")
        ax.set_title(f"Angular Rate — {accel_labels[i]}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.minorticks_on()

    if args.residuals:
        # Residuals row: noisy - input (for visual noise/ bias assessment)
        for i in range(3):
            ax = axes[2, i]
            res_a = f_meas[:, i] - input_f_B[:, i]
            res_w = w_meas[:, i] - data["gt_w_B"][:, i]
            ax.plot(t, res_a, color=colors["residual"], lw=1.0, label="residual f_B")
            ax.plot(t, res_w, color="#cc78bc", lw=1.0, alpha=0.9, label="residual w_B")
            ax.axhline(0.0, color="#666666", lw=0.8, ls="--", alpha=0.8)
            ax.set_title(f"Residuals — {accel_labels[i]}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("units")
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.minorticks_on()

    # Add a single legend per row
    handles0, labels0 = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles0, labels0, loc="upper right", frameon=True)
    handles1, labels1 = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles1, labels1, loc="upper right", frameon=True)
    if args.residuals:
        handles2, labels2 = axes[2, 0].get_legend_handles_labels()
        axes[2, 0].legend(handles2, labels2, loc="upper right", frameon=True)

    # Optional: link y-limits across rows (acc row and gyro row separately)
    if args.linky:
        def on_ylim_changed(ax_changed):
            # Determine row of the changed axis
            for ridx in range(nrows):
                for cidx in range(3):
                    if axes[ridx, cidx] is ax_changed:
                        # Sync this row's y-lims across other columns in the row
                        y0, y1 = ax_changed.get_ylim()
                        for cc in range(3):
                            if cc != cidx:
                                axes[ridx, cc].set_ylim(y0, y1)
                        return

        # Connect callbacks for each axis
        for ridx in range(nrows):
            for cidx in range(3):
                axes[ridx, cidx].callbacks.connect('ylim_changed', on_ylim_changed)

    # Keyboard shortcuts
    #  - press '0' to autoscale all y-limits per row
    def on_key(event):
        if event.key == '0':
            for ridx in range(nrows):
                for cidx in range(3):
                    axes[ridx, cidx].relim()
                    axes[ridx, cidx].autoscale_view()
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Tight layout and show interactively
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
