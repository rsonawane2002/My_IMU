import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "/Users/siddharth/Downloads/imu_sim2real_plus/runs/input_data_only"

try:
    # Load the input data
    input_a_B = np.load(os.path.join(output_dir, "seq_00000_input_a_B.npy"))
    gt_w_B = np.load(os.path.join(output_dir, "seq_00000_gt_w_B.npy"))

    # Determine time vector
    num_samples = input_a_B.shape[0]
    odr = 100  # Assuming 100 Hz ODR based on previous scripts
    time = np.arange(num_samples) / odr

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle('IMU Input Data (Linear Acceleration and Angular Velocity)', fontsize=16)

    # Plot Linear Acceleration (a_B)
    accel_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(time, input_a_B[:, i], 'r-', label='Input Linear Acceleration')
        ax.set_title(f'Linear Acceleration - Axis {accel_labels[i]}')
        ax.set_ylabel('m/s^2')
        ax.legend()
        ax.grid(True)

    # Plot Angular Velocity (w_B)
    gyro_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[1, i]
        ax.plot(time, gt_w_B[:, i], 'b-', label='Input Angular Velocity')
        ax.set_title(f'Angular Velocity - Axis {gyro_labels[i]}')
        ax.set_ylabel('rad/s')
        ax.legend()
        ax.grid(True)

    # Set common X label
    for ax in axes[1, :]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plots
    plot_filename = os.path.join(output_dir, "imu_input_data_plot.png")
    plt.savefig(plot_filename)
    print(f"Input data plots saved to: {plot_filename}")

except FileNotFoundError:
    print(f"Error: Input data files not found. Please ensure all required .npy files exist in {output_dir}.")
except Exception as e:
    print(f"An error occurred: {e}")
