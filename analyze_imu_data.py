import numpy as np

f_meas_path = "/Users/siddharth/Downloads/imu_sim2real_plus/runs/sim/seq_00000_f_meas.npy"
w_meas_path = "/Users/siddharth/Downloads/imu_sim2real_plus/runs/sim/seq_00000_w_meas.npy"

try:
    f_meas = np.load(f_meas_path)
    w_meas = np.load(w_meas_path)

    print("--- Linear Acceleration (f_meas) ---")
    print(f"Shape: {f_meas.shape}")
    print(f"Mean: {np.mean(f_meas, axis=0)}")
    print(f"Standard Deviation: {np.std(f_meas, axis=0)}")
    print(f"Min: {np.min(f_meas, axis=0)}")
    print(f"Max: {np.max(f_meas, axis=0)}")

    print("\n--- Angular Velocity (w_meas) ---")
    print(f"Shape: {w_meas.shape}")
    print(f"Mean: {np.mean(w_meas, axis=0)}")
    print(f"Standard Deviation: {np.std(w_meas, axis=0)}")
    print(f"Min: {np.min(w_meas, axis=0)}")
    print(f"Max: {np.max(w_meas, axis=0)}")

    print("\n--- Interpretation ---")
    print("Simulated Input (Ground Truth): For this specific dataset generation, the ground truth linear acceleration and angular velocity are both zero throughout the sequence.")
    print("Realistic (Noise Included) Output: The loaded data (f_meas and w_meas) represents the synthetic IMU measurements with noise applied.")
    print("If plotted, you would see the f_meas and w_meas data fluctuating around zero, demonstrating the effect of the added noise on the otherwise static (zero) ground truth signal.")

except FileNotFoundError:
    print(f"Error: One or both files not found. Make sure '{f_meas_path}' and '{w_meas_path}' exist.")
except Exception as e:
    print(f"An error occurred: {e}")
