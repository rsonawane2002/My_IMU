import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- CONFIG ---
# Path to the folder containing 'traj_0', 'traj_1', etc.
DATA_DIR = os.path.expanduser("~/Documents/trajectories_verification")

def plot_verification(traj_path):
    # Find files
    clean_files = glob.glob(os.path.join(traj_path, "clean_imu_*.csv"))
    custom_files = glob.glob(os.path.join(traj_path, "custom_imu_*.csv"))

    if not clean_files or not custom_files:
        print(f"Skipping {traj_path}: Missing CSV files.")
        return

    # Load Data
    df_clean = pd.read_csv(clean_files[0])
    df_custom = pd.read_csv(custom_files[0])

    # Normalize Time (start at 0)
    t_clean = df_clean['time'] - df_clean['time'].iloc[0]
    t_custom = df_custom['time'] - df_custom['time'].iloc[0]

    # Setup Figure (3x2 Grid)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Verification: {os.path.basename(traj_path)}", fontsize=16)

    # Plot Pairs (Col 0 = Accel, Col 1 = Gyro)
    # Format: (Title, Clean_Col, Custom_Col, Unit, Axis_Index)
    
    plots = [
        ("Accel X", "imu_ax", "imu_ax", "m/s^2", (0, 0)),
        ("Accel Y", "imu_ay", "imu_ay", "m/s^2", (1, 0)),
        ("Accel Z", "imu_az", "imu_az", "m/s^2", (2, 0)),
        ("Gyro X",  "imu_gx", "imu_gx", "rad/s", (0, 1)),
        ("Gyro Y",  "imu_gy", "imu_gy", "rad/s", (1, 1)),
        ("Gyro Z",  "imu_gz", "imu_gz", "rad/s", (2, 1)),
    ]

    for title, col_c, col_n, unit, (row, col) in plots:
        ax = axes[row, col]
        
        # Plot Custom (Noisy) FIRST so it's in the background
        ax.plot(t_custom, df_custom[col_n], 'r-', alpha=0.5, linewidth=1.0, label='Custom (C++)')
        
        # Plot Clean (Raw) SECOND so it overlays clearly on top (or dashed)
        ax.plot(t_clean, df_clean[col_c], 'k--', alpha=0.8, linewidth=1.5, label='Clean (Isaac)')
        
        ax.set_ylabel(f"{title} [{unit}]")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc='upper right', frameon=True)
    
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Plot
    out_path = os.path.join(traj_path, "verification_plot.png")
    plt.savefig(out_path)
    print(f"Generated plot: {out_path}")
    plt.close()

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory not found: {DATA_DIR}")
        return

    # Find all trajectory subfolders
    traj_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "traj_*")))
    
    if not traj_dirs:
        print("No trajectory folders found!")
        return

    print(f"Found {len(traj_dirs)} trajectories. Generating plots...")
    for traj in traj_dirs:
        plot_verification(traj)

if __name__ == "__main__":
    main()