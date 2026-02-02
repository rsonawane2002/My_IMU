import pandas as pd
import numpy as np
import allantools
import matplotlib.pyplot as plt

def compute_fast_allan(filename, fs=400):
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    
    # 1. Prepare Data
    data = {
        'Accel X': df['imu_ax'].values,
        'Accel Y': df['imu_ay'].values,
        'Accel Z': df['imu_az'].values,
        'Gyro X':  df['imu_gx'].values, 
        'Gyro Y':  df['imu_gy'].values, 
        'Gyro Z':  df['imu_gz'].values  
    }

    # 2. GENERATE TAUS MANUALLY (The Fix)
    # We create 100 points spaced logarithmically from 1/fs to 1000 seconds.
    # This prevents the computer from calculating millions of useless points.
    t_max = 2000.0  # Calculate up to 2000 seconds
    taus_list = np.logspace(np.log10(1/fs), np.log10(t_max), num=100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 3. Compute Accel
    print("Computing Accel (Fast Mode)...")
    for axis in ['Accel X', 'Accel Y', 'Accel Z']:
        # Note: We pass the explicit 'taus_list' here
        (taus_out, ad, ade, ns) = allantools.oadev(
            data[axis], rate=fs, data_type="freq", taus=taus_list
        )
        ax1.loglog(taus_out, ad, label=axis)
        
    ax1.set_title("Accelerometer Allan Deviation")
    ax1.set_xlabel("Tau (s)")
    ax1.set_ylabel("Deviation (m/s^2)")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    # 4. Compute Gyro
    print("Computing Gyro (Fast Mode)...")
    for axis in ['Gyro X', 'Gyro Y', 'Gyro Z']:
        (taus_out, ad, ade, ns) = allantools.oadev(
            data[axis], rate=fs, data_type="freq", taus=taus_list
        )
        ax2.loglog(taus_out, ad, label=axis)

    ax2.set_title("Gyroscope Allan Deviation")
    ax2.set_xlabel("Tau (s)")
    ax2.set_ylabel("Deviation (deg/s)")
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_filename = filename.replace('.csv', '_avar_fast.png')
    plt.savefig(plot_filename)
    print(f"Done! Plot saved to {plot_filename}")

if __name__ == "__main__":
    target_csv = "sim_noisy_stationary.csv" 
    compute_fast_allan(target_csv, fs=400)