import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys  # <--- ADDED

# 1. LOAD DATA
df_inference = pd.read_csv("sim2real_output.csv")

# SET TRAJECTORY ID FROM ARGS
if len(sys.argv) < 2:
    print("Usage: python plot.py <traj_id>")
    sys.exit(1)

traj_id = sys.argv[1]

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

# 2. SETUP PLOTTING
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
fig.suptitle(f"CLEAN INPUT: Sim-to-Real IMU Alignment Comparison (Traj {traj_id})", fontsize=16)

# UPDATED MAPPING: Added 'Physics_Col' at the end
# (Row, Col, Title, Original_Col, Sim2Real_Col, Real_Col, Physics_Col)
plot_config = [
    (0, 0, "Accel X", "ax_original", "ax_sim2real", "acc_x_g",      "imu_ax"),
    (0, 1, "Accel Y", "ay_original", "ay_sim2real", "acc_y_g",      "imu_ay"),
    (0, 2, "Accel Z", "az_original", "az_sim2real", "acc_z_g",      "imu_az"),
    (1, 0, "Gyro X",  "gx_original", "gx_sim2real", "gyro_x_rad_s", "imu_gx"),
    (1, 1, "Gyro Y",  "gy_original", "gy_sim2real", "gyro_y_rad_s", "imu_gy"),
    (1, 2, "Gyro Z",  "gz_original", "gz_sim2real", "gyro_z_rad_s", "imu_gz")
]

for row, col, title, sim_col, s2r_col, real_col, phys_col in plot_config:
    ax = axes[row, col]
    
    # 1. Plot Simulated Input (The "Clean" data)
    ax.plot(df_inference[sim_col], label="Sim (Original)", color='blue', alpha=0.6, linestyle='--')
    
    # 2. Plot Sim2Real Output (The "Corrected + Noisy" data)
    ax.plot(df_inference[s2r_col], label="Sim2Real (Model)", color='red', alpha=0.8)
    
    # 3. Plot Sid's Physics Pipeline (Using 'phys_col' now)
    if physics_exists and phys_col in df_physics.columns:
        ax.plot(df_physics[phys_col].values[:len(df_inference)], 
                label="Sid's Physics Pipeline", color='green', alpha=0.6, linestyle='-.')
    
    # 4. Plot Real Ground Truth (The "Target")
    if real_exists and real_col in df_real.columns:
        ax.plot(df_real[real_col].values[:len(df_inference)], 
                label="Real (Ground Truth)", color='black', alpha=0.5)

    ax.set_title(title)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()