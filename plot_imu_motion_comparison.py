
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths to the generated data
output_dir = "/Users/siddharth/Downloads/imu_sim2real_plus/runs/variable_motion_sim"

try:
    # Load the noisy measurements and ground truth motion
    f_meas = np.load(os.path.join(output_dir, "seq_00000_f_meas.npy"))
    w_meas = np.load(os.path.join(output_dir, "seq_00000_w_meas.npy"))
    gt_a_W = np.load(os.path.join(output_dir, "seq_00000_gt_a_W.npy"))
    gt_w_B = np.load(os.path.join(output_dir, "seq_00000_gt_w_B.npy"))
    gt_R_WB = np.load(os.path.join(output_dir, "seq_00000_gt_R_WB.npy"))

    # Determine time vector
    num_samples = f_meas.shape[0]
    odr = 400 # From generate_dataset command
    time = np.arange(num_samples) / odr

    # Calculate ground truth specific force (f_B = R_BW * (a_W - g_W))
    g_W = np.array([0, 0, 9.81])
    R_BW = gt_R_WB.transpose(0, 2, 1)
    a_W_minus_g = gt_a_W - g_W
    gt_f_B = np.einsum('nij,nj->ni', R_BW, a_W_minus_g)


    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle('IMU Motion Comparison', fontsize=16)

    # Plot Specific Force (Accelerometer)
    accel_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(time, gt_f_B[:, i], 'r--', label='GT Specific Force')
        ax.plot(time, f_meas[:, i], 'b-', alpha=0.7, label='Noisy Specific Force')
        ax.set_title(f'Specific Force - Axis {accel_labels[i]}')
        ax.set_ylabel('m/s^2')
        ax.legend()
        ax.grid(True)

    # Plot Angular Velocity
    gyro_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[1, i]
        ax.plot(time, gt_w_B[:, i], 'r--', label='GT')
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
    plot_filename = os.path.join(output_dir, "imu_motion_comparison_plot.png")
    plt.savefig(plot_filename)
    print(f"Plots saved to: {plot_filename}")

except FileNotFoundError:
    print(f"Error: Data files not found. Please ensure all required .npy files exist in {output_dir}.")
except Exception as e:
    print(f"An error occurred: {e}")
