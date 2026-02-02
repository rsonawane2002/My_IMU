import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.fft import fft, fftfreq

# ==========================================
# HELPER: Compute FFT
# ==========================================
def get_fft(data, fs):
    """
    Calculates the FFT of a 1D signal.
    Args:
        data: array-like 1D signal
        fs: sampling frequency (Hz)
    Returns:
        xf: frequency axis (Hz)
        yf: magnitude of the FFT
    """
    # Remove DC component (gravity/mean offset) so we just see vibrations
    data = np.array(data) - np.mean(data)
    
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / fs)
    
    # Return only the positive half of the spectrum
    return xf[:N//2], 2.0/N * np.abs(yf[0:N//2])

# ==========================================
# 1. LOAD DATA
# ==========================================
df_inference = pd.read_csv("sim2real_output.csv")

# SET TRAJECTORY ID FROM ARGS
if len(sys.argv) < 2:
    print("Usage: python plot_fft.py <traj_id>")
    sys.exit(1)

traj_id = sys.argv[1]

# SAMPLING RATE (Critical for FFT)
# Based on your filenames "real_400hz", we assume 400 Hz.
FS = 400.0 

# Load the ground truth real data
try:
    df_real = pd.read_csv(f"./logs_2/traj_{traj_id}/real_400hz_{traj_id}.csv")
    real_exists = True
except FileNotFoundError:
    print(f"Real ground truth file (ID {traj_id}) not found.")
    real_exists = False

# Load Sid's physics pipeline data
try:
    df_physics = pd.read_csv(f"./logs_2/traj_{traj_id}/sim_noisy_{traj_id}.csv")
    physics_exists = True
except FileNotFoundError:
    print(f"Physics pipeline file (ID {traj_id}) not found.")
    physics_exists = False

# ==========================================
# 2. SETUP PLOTTING
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"FFT FREQUENCY ANALYSIS: Sim-to-Real Comparison (Traj {traj_id})", fontsize=16)

# (Row, Col, Title, Original_Col, Sim2Real_Col, Real_Col, Physics_Col)
plot_config = [
    (0, 0, "Accel X FFT", "ax_original", "ax_sim2real", "acc_x_g",      "imu_ax"),
    (0, 1, "Accel Y FFT", "ay_original", "ay_sim2real", "acc_y_g",      "imu_ay"),
    (0, 2, "Accel Z FFT", "az_original", "az_sim2real", "acc_z_g",      "imu_az"),
    (1, 0, "Gyro X FFT",  "gx_original", "gx_sim2real", "gyro_x_rad_s", "imu_gx"),
    (1, 1, "Gyro Y FFT",  "gy_original", "gy_sim2real", "gyro_y_rad_s", "imu_gy"),
    (1, 2, "Gyro Z FFT",  "gz_original", "gz_sim2real", "gyro_z_rad_s", "imu_gz")
]

for row, col, title, sim_col, s2r_col, real_col, phys_col in plot_config:
    ax = axes[row, col]
    
    # 1. Plot Simulated Input (Clean)
    xf, yf = get_fft(df_inference[sim_col], FS)
    ax.plot(xf, yf, label="Sim (Original)", color='blue', alpha=0.6, linestyle='--')
    
    # 2. Plot Sim2Real Output (Model)
    xf, yf = get_fft(df_inference[s2r_col], FS)
    ax.plot(xf, yf, label="Sim2Real (Model)", color='red', alpha=0.8)
    
    # 3. Plot Sid's Physics Pipeline
    if physics_exists and phys_col in df_physics.columns:
        # Match length to inference just in case
        data_slice = df_physics[phys_col].values[:len(df_inference)]
        xf, yf = get_fft(data_slice, FS)
        ax.plot(xf, yf, label="Sid's Physics Pipeline", color='green', alpha=0.6, linestyle='-.')
    
    # 4. Plot Real Ground Truth
    if real_exists and real_col in df_real.columns:
        data_slice = df_real[real_col].values[:len(df_inference)]
        xf, yf = get_fft(data_slice, FS)
        ax.plot(xf, yf, label="Real (Ground Truth)", color='black', alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, FS / 2)  # Show up to Nyquist frequency (200Hz)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Only show legend on the first plot to avoid clutter
    if row == 0 and col == 0:
        ax.legend(loc='upper right', fontsize='small')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()