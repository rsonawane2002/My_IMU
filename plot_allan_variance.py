import numpy as np
import matplotlib.pyplot as plt
import os
from imu_sim2real_plus.metrics.allan import allan_deviation

def plot_allan_variance():
    # Define paths to the generated data
    data_dir = "runs/long_term_sim"
    f_meas_path = os.path.join(data_dir, "seq_00000_f_meas.npy")
    w_meas_path = os.path.join(data_dir, "seq_00000_w_meas.npy")
    output_plot_path = os.path.join(data_dir, "allan_variance_plot.png")

    # Load the measurements
    try:
        f_meas = np.load(f_meas_path)
        w_meas = np.load(w_meas_path)
    except FileNotFoundError:
        print(f"Error: Data files not found in {data_dir}. Please generate the long-term data first.")
        return

    # --- Parameters ---
    fs = 400  # Sampling frequency in Hz (from generate_dataset command)
    # Generate logarithmically spaced tau values
    taus = np.logspace(-1, 2, 50) # from 0.1 to 100 seconds

    # --- Calculate Allan Deviation for each axis ---
    adev_accel = np.array([allan_deviation(f_meas[:, i], fs, taus) for i in range(3)])
    adev_gyro = np.array([allan_deviation(w_meas[:, i], fs, taus) for i in range(3)])

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Allan Deviation', fontsize=16)
    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    # Accelerometer plot
    ax1 = axes[0]
    for i in range(3):
        ax1.loglog(taus, adev_accel[i, :], color=colors[i], marker='o', markersize=3, linestyle='-', label=f'Accel {labels[i]}')
    ax1.set_title('Accelerometer')
    ax1.set_xlabel('Averaging Time (τ) [s]')
    ax1.set_ylabel('Allan Deviation (σ) [m/s²]')
    ax1.grid(True, which="both", ls="-")
    ax1.legend()

    # Gyroscope plot
    ax2 = axes[1]
    for i in range(3):
        ax2.loglog(taus, adev_gyro[i, :], color=colors[i], marker='o', markersize=3, linestyle='-', label=f'Gyro {labels[i]}')
    ax2.set_title('Gyroscope')
    ax2.set_xlabel('Averaging Time (τ) [s]')
    ax2.set_ylabel('Allan Deviation (σ) [rad/s]')
    ax2.grid(True, which="both", ls="-")
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    plt.savefig(output_plot_path)
    print(f"Allan variance plot saved to: {output_plot_path}")

if __name__ == '__main__':
    plot_allan_variance()
