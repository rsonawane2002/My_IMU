import pandas as pd
import os
# Import the specific functions from your script above
# (Assuming you saved your physics code as physics_pipeline.py)
from inject_vibration_and_compare import load_clean, synth_with_motor_bands

def process_stationary_file():
    # 1. Define paths
    input_path = "clean_stationary_6h.csv"
    output_path = "sim_noisy_stationary.csv"
    
    # 2. Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        return

    print(f"Loading {input_path}...")
    
    # 3. Load and Prep
    # load_clean renames columns (imu_ax -> acc_x) and adds time norm
    df_clean = load_clean(input_path)
    
    # 4. Run Physics
    print("Running physics simulation (Motor Bands + Temp Model)...")
    df_noisy = synth_with_motor_bands(df_clean)
    
    # 5. Save
    df_noisy.to_csv(output_path, index=False)
    print(f"Success! Noisy data saved to: {output_path}")

if __name__ == "__main__":
    process_stationary_file()