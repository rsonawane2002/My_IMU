import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_single_trajectory(base_dir, traj_id):
    traj_path = os.path.join(base_dir, traj_id)
    print(f"--- Processing: {traj_path} ---")

    # 1. Robust ID Extraction
    match = re.search(r'(\d+)$', traj_id)
    if match:
        id_suffix = match.group(1)
    else:
        id_suffix = traj_id
        print(f"Warning: Could not extract number from '{traj_id}'. Using full string.")

    # Construct filenames
    sim_filename   = f"imu_400hz_{id_suffix}.csv"
    real_filename  = f"real_400hz_{id_suffix}.csv"
    noisy_filename = f"sim_noisy_{id_suffix}.csv"  # <--- NEW FILE

    # 2. Load Data
    try:
        sim_path   = os.path.join(traj_path, sim_filename)
        real_path  = os.path.join(traj_path, real_filename)
        noisy_path = os.path.join(traj_path, noisy_filename)

        print(f"Loading: {sim_filename}")
        df_sim = pd.read_csv(sim_path)
        
        print(f"Loading: {real_filename}")
        df_real = pd.read_csv(real_path)

        print(f"Loading: {noisy_filename}")
        df_noisy = pd.read_csv(noisy_path) # <--- Load Noisy Data

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: Could not find one of the files.")
        print(f"Python Error: {e}")
        return

    # 3. Define Column Mappings
    # (Row, Col, Title, Sim_Col, Real_Col)
    plots_config = [
        (0, 0, 'Accel X', 'imu_ax', 'acc_x_g'),
        (0, 1, 'Accel Y', 'imu_ay', 'acc_y_g'),
        (0, 2, 'Accel Z', 'imu_az', 'acc_z_g'),
        (1, 0, 'Gyro X',  'imu_gx', 'gyro_x_rad_s'),
        (1, 1, 'Gyro Y',  'imu_gy', 'gyro_y_rad_s'),
        (1, 2, 'Gyro Z',  'imu_gz', 'gyro_z_rad_s'),
    ]

    # 4. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(f'Comparison: {traj_id} (Clean Sim vs Noisy Sim vs Real)', fontsize=16)

    for row, col, title, sim_col, real_col in plots_config:
        ax = axes[row, col]
        
        # A. Plot Real Data (Ground Truth) - Black
        if real_col in df_real.columns:
            x_real = df_real['time'] if 'time' in df_real.columns else df_real.index
            ax.plot(x_real, df_real[real_col], label='Real (GT)', color='black', alpha=0.8, linewidth=1.5, zorder=3)
        
        # B. Plot Clean Sim Data - Orange Dashed
        if sim_col in df_sim.columns:
            x_sim = df_sim['time'] if 'time' in df_sim.columns else df_sim.index
            ax.plot(x_sim, df_sim[sim_col], label='Sim (Clean)', color='orange', linestyle='--', linewidth=1.5, zorder=2)

        # C. Plot Noisy Sim Data - Red (NEW)
        # We reuse 'sim_col' because the generated file uses the same column names as clean sim
        if sim_col in df_noisy.columns:
            x_noisy = df_noisy['time'] if 'time' in df_noisy.columns else df_noisy.index
            ax.plot(x_noisy, df_noisy[sim_col], label='Sim (Noisy)', color='red', alpha=0.5, linewidth=1, zorder=1)

        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Legend only on first subplot
        if row == 0 and col == 0:
            ax.legend()
    
    axes[1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    LOGS_DIR = 'new_pipeline'
    
    # Ensure this matches your folder name exactly
    TARGET_TRAJ = 'traj_251' 
    
    plot_single_trajectory(LOGS_DIR, TARGET_TRAJ)