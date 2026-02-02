import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# --- Import your Python Physics Engine ---
# Ensure your python path can find 'imu_sim2real_plus'
sys.path.append(os.getcwd()) 

# Use the existing library function
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
from imu_vibration_sim import ResonantMode

# --- RE-IMPLEMENTING THE PYTHON REFERENCE WRAPPER LOCALLY ---
def run_python_reference_engine(df_clean):
    # 1. Prep Inputs
    t = df_clean['time'].to_numpy()
    
    # Force dt to 104Hz to match C++ internal logic
    # (The C++ engine now forces 104Hz via the odr_hz knob)
    dt = 1.0 / 104.0 
    
    N = len(t)
    f_B_input = df_clean[['imu_ax', 'imu_ay', 'imu_az']].to_numpy() 
    w_B_input = df_clean[['imu_gx', 'imu_gy', 'imu_gz']].to_numpy()

    # 2. Mock Kinematics
    R_BW = np.repeat(np.eye(3)[np.newaxis, :, :], N, axis=0) 
    g_W = np.array([0.0, 0.0, 9.80665])
    a_W = f_B_input + g_W 
    wdot_B = np.gradient(w_B_input, dt, axis=0)
    r_lever = np.zeros(3)

    # 3. Physics Config (UPDATED TO MATCH NEW C++ DEFAULTS)
    
    # "Quiet" Vibration Modes (Matches C++ Constructor)
    accel_modes_cfg = [
        {'f0': 75.0,  'zeta': 0.02, 'gain': 0.05, 'axes': [0, 1, 2]},
        {'f0': 95.0,  'zeta': 0.03, 'gain': 0.02, 'axes': [0, 1, 2]},
        {'f0': 120.0, 'zeta': 0.03, 'gain': 0.04, 'axes': [2]},
        {'f0': 130.0, 'zeta': 0.03, 'gain': 0.03, 'axes': [2]},
    ]
    gyro_modes_cfg = [
        {'f0': 6.5,   'zeta': 0.07, 'gain': 0.02,  'axes': [0, 1, 2]},
        {'f0': 8.0,   'zeta': 0.08, 'gain': 0.015, 'axes': [0, 1, 2]},
        {'f0': 75.0,  'zeta': 0.03, 'gain': 0.004, 'axes': [2]},
        {'f0': 120.0, 'zeta': 0.03, 'gain': 0.004, 'axes': [2]},
    ]

    cfg = {
        'imu': {
            'quantization_bits': 16, 
            'accel_fs_g': 2.0,      # UPDATED: Matches C++ 2.0g
            'gyro_fs_dps': 250.0,   # UPDATED: Matches C++ 250dps
            'misalignment_pct': [0.0, 0.0], 
            'accel': {
                'scale_ppm': [0, 0],
                'bias_init': [0.0, 0.0],
                'bias_tau_s': [3600, 3600],
                # UPDATED: Datasheet Density (0.0005884)
                'noise_density': [0.0005884, 0.0005884] 
            },
            'gyro': {
                'scale_ppm': [0, 0],
                'bias_init': [0.0, 0.0],
                'bias_tau_s': [3600, 3600],
                # UPDATED: Datasheet Density (8.7266e-05)
                'noise_density': [8.7266e-05, 8.7266e-05]
            }
        },
        'vibration': {
            'g_sensitivity': 0.002,
            'motor_harmonics': {1: 1.0, 2: 0.35, 3: 0.2}, 
            'floor_noise_sigma': 0.4, 
            'floor_noise_ar': 0.96,
            'floor_noise_ma': 0.2,
            'accel_modes': accel_modes_cfg,
            'gyro_modes': gyro_modes_cfg
        }
    }

    rpm_profile = np.full_like(t, 75.0 * 60.0)

    # 4. Run Physics Simulation
    sim_out, _ = synth_measurements(
        R_WB=R_BW, 
        w_B=w_B_input,
        a_W=a_W,
        wdot_B=wdot_B,
        r_lever=r_lever,
        cfg=cfg,
        dt=dt,
        seed=123, # SEED MUST MATCH C++
        rpm_profile=rpm_profile
    )

    return pd.DataFrame({
        'time': t,
        'imu_ax': sim_out['f_meas'][:, 0],
        'imu_ay': sim_out['f_meas'][:, 1],
        'imu_az': sim_out['f_meas'][:, 2],
        'imu_gx': sim_out['w_meas'][:, 0],
        'imu_gy': sim_out['w_meas'][:, 1],
        'imu_gz': sim_out['w_meas'][:, 2],
    })

# --- CONFIG ---
DATA_DIR = os.path.expanduser("~/Documents/trajectories_verification")

def plot_comparison(traj_path, df_cpp, df_python):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Engine Validation: {os.path.basename(traj_path)}\n(Red = C++, Blue = Python Reference)", fontsize=14)

    t = df_cpp['time'] - df_cpp['time'].iloc[0]

    plots = [
        ("Accel X", "imu_ax", "m/s^2", (0, 0)),
        ("Accel Y", "imu_ay", "m/s^2", (1, 0)),
        ("Accel Z", "imu_az", "m/s^2", (2, 0)),
        ("Gyro X",  "imu_gx", "rad/s", (0, 1)),
        ("Gyro Y",  "imu_gy", "rad/s", (1, 1)),
        ("Gyro Z",  "imu_gz", "rad/s", (2, 1)),
    ]

    for title, col, unit, (row, c) in plots:
        ax = axes[row, c]
        ax.plot(t, df_cpp[col], 'r-', alpha=0.6, linewidth=1.0, label='C++ Output')
        ax.plot(t, df_python[col], 'b--', alpha=0.6, linewidth=1.0, label='Python Ref')
        ax.set_ylabel(f"{title} [{unit}]")
        ax.grid(True, alpha=0.3)
        if row == 0: ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(traj_path, "cpp_vs_python_match.png")
    plt.savefig(out_path)
    print(f"  -> Plot saved: {out_path}")
    plt.close()

def main():
    traj_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "traj_*")))
    
    if not traj_dirs:
        print("No verification trajectories found.")
        return

    print(f"Found {len(traj_dirs)} trajectories to validate.")

    for traj in traj_dirs:
        print(f"\nProcessing {os.path.basename(traj)}...")
        
        # Load Files
        clean_file = glob.glob(os.path.join(traj, "clean_imu_*.csv"))[0]
        cpp_file = glob.glob(os.path.join(traj, "custom_imu_*.csv"))[0]
        
        df_clean = pd.read_csv(clean_file)
        df_cpp = pd.read_csv(cpp_file)

        # 1. GENERATE PYTHON REFERENCE
        # Pass the CLEAN data through Python engine to get what the output "should" be
        df_python = run_python_reference_engine(df_clean)

        # 2. COMPARE STATISTICS
        # Calculate error for Accel X as a proxy
        diff = df_cpp['imu_ax'] - df_python['imu_ax']
        mae = np.mean(np.abs(diff))
        corr = np.corrcoef(df_cpp['imu_ax'], df_python['imu_ax'])[0,1]

        print(f"  -> Accel X Correlation: {corr:.5f}")
        print(f"  -> Accel X MAE: {mae:.5f}")

        # 3. PLOT OVERLAY
        plot_comparison(traj, df_cpp, df_python)

if __name__ == "__main__":
    main()