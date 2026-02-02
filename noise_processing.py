import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = './logs_2'
CUTOFF_FREQ = 15.0  # Hz (Motion is usually < 10Hz, Vibration > 20Hz)
SAMPLING_RATE = 400.0 # Hz (Approximate, script will calc exact)
FILTER_ORDER = 4

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt provides zero-phase filtering (no time delay)
    y = filtfilt(b, a, data, axis=0)
    return y

def visualize_filtering(data_dir):
    # 1. Find a valid pair
    search_pattern = os.path.join(data_dir, "**", "imu_*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    sim_path, real_path = None, None
    for f in files:
        if "real_" in os.path.basename(f): continue
        fid = os.path.basename(f).replace("imu_", "").replace(".csv", "")
        r_path = os.path.join(os.path.dirname(f), f"real_imu_{fid}.csv")
        if os.path.exists(r_path):
            sim_path = f
            real_path = r_path
            break
            
    if not sim_path:
        print("❌ No pairs found.")
        return

    print(f"Analyzing:\n  Sim: {sim_path}\n  Real: {real_path}")

    # 2. Load Data
    sim_df = pd.read_csv(sim_path)
    real_df = pd.read_csv(real_path)

    # 3. Time Alignment
    # Normalize sim time
    sim_t = sim_df['time'].values
    sim_t = sim_t - sim_t[0]

    # Normalize real time (Handle Unix)
    if 'time' in real_df.columns: t_col = 'time'
    elif 'time_sec' in real_df.columns: t_col = 'time_sec'
    else: return
    
    real_t = real_df[t_col].values
    real_t = real_t - real_t[0]

    # 4. Extract Signals (Focus on Acc Z and Gyro Y as examples)
    # Sim headers: imu_az, imu_gy
    # Real headers: acc_z_g, gyro_y_rad_s
    
    sim_acc = sim_df['imu_az'].values
    real_acc_raw = real_df['acc_z_g'].values
    
    # 5. Apply Low Pass Filter to Real Data
    # Calculate exact sampling rate of real data
    dt_real = np.mean(np.diff(real_t))
    fs_real = 1.0 / dt_real
    print(f"Detected Real Sampling Rate: {fs_real:.2f} Hz")

    real_acc_filtered = butter_lowpass_filter(real_acc_raw, CUTOFF_FREQ, fs_real, FILTER_ORDER)

    # 6. Interpolate Sim to match Real (for plotting overlap)
    # We only plot the overlapping region
    limit = min(sim_t[-1], real_t[-1])
    mask_real = real_t <= limit
    mask_sim = sim_t <= limit
    
    # 7. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot 1: The Raw Reality vs Simulation
    plt.subplot(1, 2, 1)
    plt.title(f"Raw Real vs Sim (Acc Z)\nNoisy & Vibrating")
    plt.plot(real_t[mask_real], real_acc_raw[mask_real], 'lightgray', label='Real (Raw)', alpha=0.7)
    plt.plot(sim_t[mask_sim], sim_acc[mask_sim], 'r--', label='Sim (Ideal)', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: The Filtered Reality vs Simulation
    plt.subplot(1, 2, 2)
    plt.title(f"Filtered Real vs Sim (Acc Z)\nCutoff: {CUTOFF_FREQ}Hz")
    plt.plot(real_t[mask_real], real_acc_raw[mask_real], 'lightgray', label='Real (Raw)', alpha=0.5)
    plt.plot(real_t[mask_real], real_acc_filtered[mask_real], 'b-', label='Real (Filtered)', linewidth=2)
    plt.plot(sim_t[mask_sim], sim_acc[mask_sim], 'r--', label='Sim (Ideal)', linewidth=2)
    
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_filtering(DATA_DIR)