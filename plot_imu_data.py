
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths to the generated data
output_dir = "/Users/siddharth/Downloads/imu_sim2real_plus/runs/sim"
f_meas_path = os.path.join(output_dir, "seq_00000_f_meas.npy")
w_meas_path = os.path.join(output_dir, "seq_00000_w_meas.npy")

try:
    # Load the noisy measurements
    f_meas = np.load(f_meas_path)
    w_meas = np.load(w_meas_path)

    # Determine time vector
    num_samples = f_meas.shape[0]
    odr = 400 # From generate_dataset command
    time = np.arange(num_samples) / odr

    # Define simulated input (ground truth)
    # For linear acceleration, gravity is +9.81 m/s^2 on Z-axis
    # For angular velocity, ground truth is zero
    gt_f_meas = np.zeros_like(f_meas)
    gt_f_meas[:, 2] = -9.81 # Z-axis is aligned with gravity
    gt_w_meas = np.zeros_like(w_meas)

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle('Stationary IMU Data Comparison', fontsize=16)

    # Plot Specific Force (Accelerometer)
    accel_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(time, gt_f_meas[:, i], 'r--', label='GT')
        ax.plot(time, f_meas[:, i], 'b-', alpha=0.7, label='Noisy')
        ax.set_title(f'Specific Force - Axis {accel_labels[i]}')
        ax.set_ylabel('m/s^2')
        ax.legend()
        ax.grid(True)

    # Plot Angular Velocity
    gyro_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[1, i]
        ax.plot(time, gt_w_meas[:, i], 'r--', label='GT')
        ax.plot(time, w_meas[:, i], 'b-', alpha=0.7, label='Noisy')
        ax.set_title(f'Angular Velocity - Axis {gyro_labels[i]}')
        ax.set_ylabel('rad/s')
        ax.legend()
        ax.grid(True)

    # Set common X label
    for ax in axes[1, :]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plots
    plot_filename = os.path.join(output_dir, "imu_stationary_comparison_plot.png")
    plt.savefig(plot_filename)
    print(f"Plots saved to: {plot_filename}")

except FileNotFoundError:
    print(f"Error: Data files not found. Please ensure '{f_meas_path}' and '{w_meas_path}' exist.")
except Exception as e:
    print(f"An error occurred: {e}")
