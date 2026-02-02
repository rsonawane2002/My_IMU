import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# Configuration
DATA_DIR = './logs_2'
CUTOFF_FREQ = 15.0  # Hz
ORDER = 4
TARGET_COLS = ['acc_x_g', 'acc_y_g', 'acc_z_g', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']

def low_pass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(f"Cutoff ({cutoff} Hz) >= Nyquist ({nyq:.2f} Hz). Sampling rate too low.")
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "real_imu_*.csv"), recursive=True))
    print(f"Hari was here")
    print(f"Found {len(files)} files to preprocess...")
    
    saved_count = 0
    bad_files = []

    for file_path in tqdm(files):
        # Construct the potential output path so we can delete it if things go wrong
        dirname = os.path.dirname(file_path)
        filename_out = os.path.basename(file_path).replace("real_imu_", "real_filtered_imu_")
        save_path = os.path.join(dirname, filename_out)

        error_msg = None

        try:
            df = pd.read_csv(file_path)

            if 'time' not in df.columns:
                raise ValueError("No 'time' column")
            
            # 1. Fix Duplicates
            df = df.drop_duplicates(subset=['time']).reset_index(drop=True)
            time_vals = df['time'].values
            
            if len(time_vals) < 10:
                raise ValueError("Too few data points (<10)")

            # 2. Robust FS Calculation
            dt = np.mean(np.diff(time_vals))
            if dt <= 1e-5:
                raise ValueError(f"dt too small or zero ({dt})")
            
            fs = 1.0 / dt
            
            # 3. Nyquist Check
            if fs <= 2 * CUTOFF_FREQ:
                raise ValueError(f"Low sampling rate ({fs:.1f} Hz)")

            # 4. Filter
            filtered_df = df.copy()
            cols_to_filter = [c for c in TARGET_COLS if c in df.columns]
            
            if cols_to_filter:
                filtered_data = low_pass_filter(df[cols_to_filter].values, fs, CUTOFF_FREQ, ORDER)
                
                # Check for NaNs
                if np.isnan(filtered_data).any():
                    raise ValueError("Filtering generated NaNs")
                    
                filtered_df[cols_to_filter] = filtered_data
                
                # 5. Save (Success)
                filtered_df.to_csv(save_path, index=False)
                saved_count += 1
            else:
                # If no target columns, just copy raw (or skip, but safe to copy)
                filtered_df.to_csv(save_path, index=False)
                saved_count += 1

        except Exception as e:
            error_msg = str(e)
            
        # --- CLEANUP LOGIC ---
        if error_msg:
            # Add to bad list
            bad_files.append(f"{file_path}  ({error_msg})")
            
            # DELETE corrupted file if it exists from a previous run
            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"🗑️  Deleted corrupted file: {save_path}")
            else:
                print(f"⚠️  Skipping {file_path}: {error_msg}")

    print("-" * 60)
    print(f"Processing Complete.")
    print(f"✅ Successfully Saved: {saved_count}")
    print(f"❌ Failed/Skipped: {len(bad_files)}")
    print("-" * 60)
    
    if bad_files:
        print("LIST OF BAD FILES (Cleaned up):")
        for f in bad_files:
            print(f" • {f}")
    print("-" * 60)

if __name__ == "__main__":
    main()