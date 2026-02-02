import pandas as pd
import numpy as np

def generate_clean_stationary_input(
    filename="clean_stationary_24h.csv", 
    duration_h=24, 
    fs=400,
    gravity_val=-9.8
):
    print(f"Generating {duration_h} hours of CLEAN data at {fs} Hz...")
    
    # 1. Calculate size
    # 24 hours * 3600 sec/h * 400 Hz = 34,560,000 samples
    num_samples = int(duration_h * 3600 * fs)
    
    # 2. Create Time Vector
    t = np.linspace(0, duration_h * 3600, num_samples, endpoint=False, dtype=np.float32)
    
    # 3. Create DataFrame with Constant Values (No Noise)
    # Using float32 to keep memory usage lower (~1GB RAM)
    df = pd.DataFrame({
        'time': t,
        'imu_ax': np.full(num_samples, gravity_val, dtype=np.float32), # Constant Gravity
        'imu_ay': np.zeros(num_samples, dtype=np.float32),
        'imu_az': np.zeros(num_samples, dtype=np.float32),
        'imu_gx': np.zeros(num_samples, dtype=np.float32),
        'imu_gy': np.zeros(num_samples, dtype=np.float32),
        'imu_gz': np.zeros(num_samples, dtype=np.float32)
    })
    
    # 4. Export
    print(f"DataFrame created. Memory usage: ~{df.memory_usage().sum() / 1e6:.1f} MB")
    print("Writing to CSV (this will be large ~2.5GB)...")
    df.to_csv(filename, index=False)
    print(f"Done! Saved to {filename}")

if __name__ == "__main__":
    generate_clean_stationary_input()