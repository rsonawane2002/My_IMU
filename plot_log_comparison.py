import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json

parser = argparse.ArgumentParser(description="Compare input vs. realistic IMU model output from a sequence directory")
parser.add_argument("--output_dir", default="runs/from_log_new", help="Directory containing seq_00000_*.npy")
parser.add_argument("--seq", type=int, default=0, help="Sequence index (default: 0)")
args = parser.parse_args()

output_dir = args.output_dir
seq = args.seq
base = os.path.join(output_dir, f"seq_{seq:05d}")

try:
    # Load the noisy measurements and ground truth motion
    f_meas = np.load(base + "_f_meas.npy")
    w_meas = np.load(base + "_w_meas.npy")
    gt_a_W = np.load(base + "_gt_a_W.npy")
    gt_w_B = np.load(base + "_gt_w_B.npy")
    gt_R_WB = np.load(base + "_gt_R_WB.npy")
    input_a_B = np.load(base + "_input_a_B.npy")

    # Determine time vector using metadata if available
    meta_path = base + "_meta.json"
    odr = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            odr = meta.get('odr', None)
            if odr is None:
                seconds = meta.get('seconds', None)
                if seconds:
                    odr = f_meas.shape[0] / seconds
    if odr is None:
        odr = 100
    time = np.arange(f_meas.shape[0]) / float(odr)

    # Calculate ground truth specific force using NED gravity direction
    # f_B = R_WB * (a_W - g_W)
    g_W = np.array([0, 0, 9.81])
    a_W_minus_g = gt_a_W - g_W
    gt_f_B = np.einsum('nij,nj->ni', gt_R_WB, a_W_minus_g)

    # Input specific force from input a_B: f_B = a_B - (R_WB @ g_W)
    input_specific_force_B = input_a_B - np.matmul(gt_R_WB, g_W)


    # Create plots (2 rows: specific force and angular velocity)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle('IMU Data Comparison (Input vs. Realistic IMU Output)', fontsize=16)

    # Plot Specific Force (Accelerometer) - Original Comparison
    accel_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(time, input_specific_force_B[:, i], 'r--', label='Input Specific Force (Derived)')
        ax.plot(time, f_meas[:, i], 'b-', alpha=0.7, label='IMU Model Output (Realistic)')
        ax.set_title(f'Specific Force - Axis {accel_labels[i]}')
        ax.set_ylabel('m/s^2')
        ax.legend()
        ax.grid(True)

    # Plot Angular Velocity - Original Comparison
    gyro_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[1, i]
        ax.plot(time, gt_w_B[:, i], 'r--', label='Input Angular Velocity')
        ax.plot(time, w_meas[:, i], 'b-', alpha=0.7, label='IMU Model Output (Realistic)')
        ax.set_title(f'Angular Velocity - Axis {gyro_labels[i]}')
        ax.set_ylabel('rad/s')
        ax.legend()
        ax.grid(True)

    # Set common X label
    for ax in axes[1, :]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plots
    plot_filename = os.path.join(output_dir, "imu_log_comparison_plot_v2_no_acc_world_body.png")
    plt.savefig(plot_filename)
    print(f"Plots saved to: {plot_filename}")
    plt.show()

except FileNotFoundError:
    print(f"Error: Data files not found. Please ensure all required files exist in {output_dir} for seq index {seq}.")
except Exception as e:
    print(f"An error occurred: {e}")
