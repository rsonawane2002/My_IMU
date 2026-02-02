import pandas as pd
import numpy as np
import glob
import os

# Path pattern to find your files
SEARCH_PATTERN = "logs/traj_*/real_imu_*.csv"

def process_imu_logs():
    files = glob.glob(SEARCH_PATTERN)
    files.sort() # Ensure we process in order
    
    if not files:
        print(f"No files found matching: {SEARCH_PATTERN}")
        return

    print(f"Found {len(files)} IMU log files. Analyzing and Fixing...")
    print("-" * 60)
    print(f"{'Filename':<40} | {'Samples':<10} | {'Calc Hz':<10}")
    print("-" * 60)

    for filepath in files:
        try:
            # Read the CSV
            df = pd.read_csv(filepath)
            
            if df.empty or len(df) < 2:
                print(f"{os.path.basename(filepath):<40} | SKIPPING (Empty/Too short)")
                continue
            
            # 1. Get the Trustable Constraints
            # We trust the robot's clock for the Total Duration.
            # We trust the Start Time.
            t_start = df['time_sec'].iloc[0]
            t_end = df['time_sec'].iloc[-1]
            total_duration = t_end - t_start
            
            n_samples = len(df)
            
            # 2. Calculate the Effective Sampling Rate (ODR)
            # Prevent division by zero if duration is somehow 0
            if total_duration <= 0:
                print(f"{os.path.basename(filepath):<40} | ERROR (Duration <= 0)")
                continue

            calculated_hz = n_samples / total_duration
            
            # 3. Generate Perfect Linear Timestamps
            # linspace generates 'n_samples' evenly spaced numbers between start and end
            new_timestamps = np.linspace(t_start, t_end, n_samples)
            
            # 4. Overwrite
            df['time_sec'] = new_timestamps
            
            # 5. Save back to CSV
            # float_format='%.6f' ensures microsecond precision is kept
            df.to_csv(filepath, index=False, float_format='%.6f')
            
            print(f"{os.path.basename(filepath):<40} | {n_samples:<10} | {calculated_hz:.1f} Hz")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print("-" * 60)
    print("Processing complete. Timestamps have been linearized.")

if __name__ == "__main__":
    process_imu_logs()