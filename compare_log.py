import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
sim_df = pd.read_csv('logs_2/traj_300/imu_400hz_300.csv')
real_df = pd.read_csv('logs_2/traj_300/real_imu_400hz_300.csv')

# Clean up any stray spaces in column names
sim_df.columns = sim_df.columns.str.strip()
real_df.columns = real_df.columns.str.strip()

# Columns of interest
sim_cols = ['time', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
real_cols = ['time', 'acc_x_g', 'acc_y_g', 'acc_z_g', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']

# Convert to numeric (in case some entries are strings or have commas)
for col in sim_cols + real_cols:
    if col in sim_df:
        sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
    if col in real_df:
        real_df[col] = pd.to_numeric(real_df[col], errors='coerce')

# Extract time columns
sim_time = sim_df['time'].to_numpy()
real_time = real_df['time'].to_numpy()

# ✅ Normalize time (start both at t = 0)
sim_time = sim_time - sim_time[0]
real_time = real_time - real_time[0]

# Create a 3x2 grid of subplots (6 plots total)
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot accelerometer data
for i, (col_sim, col_real, label) in enumerate(zip(sim_cols[1:4], real_cols[1:4], ['ax', 'ay', 'az'])):
    axes[i].plot(sim_time, sim_df[col_sim].to_numpy(), label='Simulated ' + label)
    axes[i].plot(real_time, real_df[col_real].to_numpy(), label='Real ' + label)
    axes[i].set_title(f'IMU {label.upper()} Comparison')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Acceleration [g]')
    axes[i].legend()
    axes[i].grid(True)

# Plot gyroscope data
for i, (col_sim, col_real, label) in enumerate(zip(sim_cols[4:], real_cols[4:], ['gx', 'gy', 'gz'])):
    axes[i + 3].plot(sim_time, sim_df[col_sim].to_numpy(), label='Simulated ' + label)
    axes[i + 3].plot(real_time, real_df[col_real].to_numpy(), label='Real ' + label)
    axes[i + 3].set_title(f'IMU {label.upper()} Comparison')
    axes[i + 3].set_xlabel('Time [s]')
    axes[i + 3].set_ylabel('Angular Velocity [rad/s]')
    axes[i + 3].legend()
    axes[i + 3].grid(True)

# Adjust layout so everything fits in one window
plt.tight_layout()
plt.show()
