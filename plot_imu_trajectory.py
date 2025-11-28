import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R_scipy

# Define paths to the generated data
output_dir = "/Users/siddharth/Downloads/imu_sim2real_plus/runs/variable_motion_sim"

try:
    # Load the noisy measurements and ground truth motion from separate .npy files
    f_meas = np.load(os.path.join(output_dir, "seq_00000_f_meas.npy"))
    w_meas = np.load(os.path.join(output_dir, "seq_00000_w_meas.npy"))
    gt_R_WB = np.load(os.path.join(output_dir, "seq_00000_gt_R_WB.npy"))
    gt_w_B = np.load(os.path.join(output_dir, "seq_00000_gt_w_B.npy"))
    gt_a_W = np.load(os.path.join(output_dir, "seq_00000_gt_a_W.npy"))

    # Determine time vector
    num_samples = f_meas.shape[0]
    odr = 100 # From generate_dataset command
    dt = 1.0 / odr
    time = np.arange(num_samples) * dt

    # --- Trajectory Estimation from Noisy Measurements ---
    # Initialize estimated states
    est_position = np.zeros((num_samples, 3))
    est_velocity = np.zeros((num_samples, 3))
    est_orientation = R_scipy.from_euler('xyz', [0, 0, 0], degrees=False) # Initial orientation
    est_orientations = np.zeros((num_samples, 3, 3)) # To store rotation matrices
    est_orientations[0] = est_orientation.as_matrix()

    gravity_W = np.array([0, 0, 9.80665]) # Gravity in World frame (positive Z-up)

    for k in range(1, num_samples):
        # Integrate angular velocity to get orientation
        delta_angle = w_meas[k-1] * dt
        delta_rotation = R_scipy.from_rotvec(delta_angle)
        est_orientation = est_orientation * delta_rotation
        est_orientations[k] = est_orientation.as_matrix()

        # Transform measured acceleration from Body to World frame
        # f_meas is in body frame and includes gravity
        # We need to remove gravity to get linear acceleration only
        # accel_B = f_meas[k-1] # Measured acceleration in body frame
        # Rotate gravity vector from World to Body frame and subtract it from f_meas
        gravity_B = est_orientation.inv().apply(gravity_W) # Gravity vector in body frame
        linear_accel_B = f_meas[k-1] - gravity_B # Linear acceleration in body frame

        # Transform linear acceleration from Body to World frame
        linear_accel_W = est_orientation.apply(linear_accel_B)

        # Integrate acceleration to get velocity
        est_velocity[k] = est_velocity[k-1] + linear_accel_W * dt

        # Integrate velocity to get position
        est_position[k] = est_position[k-1] + est_velocity[k] * dt

    # --- Plotting Trajectories ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Ground Truth Trajectory (from integrated gt_a_W)
    # Need to integrate gt_a_W twice to get position
    gt_velocity = np.zeros((num_samples, 3))
    gt_position = np.zeros((num_samples, 3))
    for k in range(1, num_samples):
        gt_velocity[k] = gt_velocity[k-1] + gt_a_W[k-1] * dt
        gt_position[k] = gt_position[k-1] + gt_velocity[k] * dt

    ax.plot(gt_position[:, 0], gt_position[:, 1], gt_position[:, 2], label='Ground Truth Trajectory', color='blue', linestyle='--')
    ax.plot(est_position[:, 0], est_position[:, 1], est_position[:, 2], label='Estimated Trajectory (Noisy IMU)', color='red')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('IMU Trajectory Comparison')
    ax.legend()
    ax.grid(True)

    # Save the plot
    plot_filename = os.path.join(output_dir, "imu_trajectory_comparison_plot.png")
    plt.savefig(plot_filename)
    print(f"Trajectory plots saved to: {plot_filename}")

except FileNotFoundError:
    print(f"Error: Data files not found. Please ensure all required .npy files exist in {output_dir}.")
except Exception as e:
    print(f"An error occurred: {e}")