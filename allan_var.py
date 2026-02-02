import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import allantools

 

def analyze_large_imu_complete(csv_file_path):

    print(f"1. Reading {csv_file_path}...")

   

    # Load data

    df = pd.read_csv(csv_file_path)

   

    # Calculate fs

    dt = np.mean(np.diff(df['time']))

    fs = 1.0 / dt

    N = len(df)

    print(f"   Loaded {N} samples. Sampling Rate: {fs:.2f} Hz")

   

    # --- Unit Conversions ---

    print("2. Converting units...")

    # Gyro: rad/s -> deg/s

    cols_gyro = ['imu_gx', 'imu_gy', 'imu_gz']

    for c in cols_gyro:

        df[c] *= (180.0 / np.pi)

 

    # Accel: g -> m/s^2

    cols_accel = ['imu_ax', 'imu_ay', 'imu_az']

    for c in cols_accel:

        df[c] *= 9.80665

       

    # --- Optimization: Log-Spaced Taus ---

    # We calculate ~100 points spread logarithmically.

    # This prevents the script from freezing on 1.4GB of data.

    max_tau = (N / fs) / 3.0

    taus_list = np.logspace(np.log10(1/fs), np.log10(max_tau), 100)

   

    print(f"3. Calculating Allan Deviation (using {len(taus_list)} tau points)...")

 

    # Setup Plots

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   

    # --- GYROSCOPE ANALYSIS ---

    print("\n--- Gyroscope Metrics ---")

    ax1.set_title("Gyroscope Allan Deviation")

    ax1.set_xlabel("Tau (s)")

    ax1.set_ylabel("Allan Deviation (deg/s)")

    ax1.grid(True, which="both", ls="-", alpha=0.5)

   

    for col in cols_gyro:

        (taus, adev, errors, ns) = allantools.oadev(

            df[col].values, rate=fs, data_type='freq', taus=taus_list

        )

       

        # 1. Bias Stability (Minimum point)

        bs = np.min(adev)

       

        # 2. Angle Random Walk (ARW) -> Value at tau=1.0

        arw = np.interp(1.0, taus, adev)

       

        print(f"[{col}]")

        print(f"   Bias Stability: {bs:.5f} deg/s")

        print(f"   ARW:            {arw:.5f} deg/√s")

       

        ax1.loglog(taus, adev, label=col)

    ax1.legend()

 

    # --- ACCELEROMETER ANALYSIS ---

    print("\n--- Accelerometer Metrics ---")

    ax2.set_title("Accelerometer Allan Deviation")

    ax2.set_xlabel("Tau (s)")

    ax2.set_ylabel("Allan Deviation (m/s^2)")

    ax2.grid(True, which="both", ls="-", alpha=0.5)

   

    for col in cols_accel:

        (taus, adev, errors, ns) = allantools.oadev(

            df[col].values, rate=fs, data_type='freq', taus=taus_list

        )

       

        # 1. Bias Stability

        bs = np.min(adev)

       

        # 2. Velocity Random Walk (VRW) / Accel Noise -> Value at tau=1.0

        vrw = np.interp(1.0, taus, adev)

       

        print(f"[{col}]")

        print(f"   Bias Stability: {bs:.5f} m/s^2")

        print(f"   VRW (Acc Noise):{vrw:.5f} m/s^2/√s")

       

        ax2.loglog(taus, adev, label=col)

    ax2.legend()

   

    plt.tight_layout()

    print("\n4. Done! Displaying plots...")

    plt.show()

 

analyze_large_imu_complete('noisy_stationary_24h.csv')