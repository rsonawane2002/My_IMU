import os
import glob
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from imu_sim2real_plus.sensors.imu_synth import synth_measurements

from imu_vibration_sim import (
    ResonantMode,
    simulate_imu_with_vibration,
)

# Reuse helpers from plotting script for consistency
from plot_all_logs_imu import (
    load_and_prepare,
    plot_pair,
    compute_axis_alignment_separate,
    apply_axis_alignment_separate,
)

def load_clean(clean_csv: str) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)
    df.columns = df.columns.str.strip()
    
    # Map your specific CSV columns to the script's internal variable names
    col_map = {
        'imu_ax': 'acc_x',
        'imu_ay': 'acc_y',
        'imu_az': 'acc_z',
        'imu_gx': 'ang_vel_x',
        'imu_gy': 'ang_vel_y',
        'imu_gz': 'ang_vel_z',
        'time':   'time'
    }
    
    # Validation
    miss = [k for k in col_map.keys() if k not in df.columns]
    if miss:
        raise ValueError(f"{os.path.basename(clean_csv)} missing columns: {miss}")
    
    # Rename columns to standard internal names
    df = df.rename(columns=col_map)
    
    df['t_norm'] = df['time'] - df['time'].iloc[0]
    return df

def synth_with_motor_bands(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Prep Inputs
    t = df['time'].to_numpy()
    
    # FORCE 104Hz dt calculation to match Config/Datasheet ODR
    dt = 1.0 / 104.0 
    
    N = len(t)
    f_B_input = df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    w_B_input = df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].to_numpy()

    # 2. Mock Kinematics for Physics Engine
    R_BW = np.repeat(np.eye(3)[np.newaxis, :, :], N, axis=0) 
    g_W = np.array([0.0, 0.0, 9.80665])
    a_W = f_B_input + g_W 
    wdot_B = np.gradient(w_B_input, dt, axis=0)
    r_lever = np.zeros(3)

    # 3. Physics Config (MATCHING DATASHEET_TYP.YAML)
    cfg = {
        'imu': {
            'quantization_bits': 16, 
            'accel_fs_g': 2.0,       # ±2 g
            'gyro_fs_dps': 250.0,    # ±250 dps
            'misalignment_pct': [-1.0, 1.0], 
            'accel': {
                'scale_ppm': [-3000, 3000],
                'bias_init': [-0.00981, 0.00981], # ±1 mg
                'bias_tau_s': [200, 2000],
                # Acceleration noise density (typ): 60 µg/√Hz
                'noise_density': [0.0005884, 0.0005884] 
            },
            'gyro': {
                'scale_ppm': [-1000, 1000],
                'bias_init': [-0.001, 0.001],    # ~±0.057 deg/s
                'bias_tau_s': [200, 2000],
                # Gyro rate noise density typ: 5 mdps/√Hz
                'noise_density': [8.7266e-05, 8.7266e-05]
            }
        },
        # "Intentionally omit vibration to reflect datasheet-like stationary conditions"
        'vibration': None 
    }

    # 4. Run Physics Simulation
    # Pass zero RPM since vibration is disabled
    rpm_profile = np.zeros_like(t)

    sim_out, _ = synth_measurements(
        R_WB=R_BW, 
        w_B=w_B_input,
        a_W=a_W,
        wdot_B=wdot_B,
        r_lever=r_lever,
        cfg=cfg,
        dt=dt,
        seed=42, # Matches 'dataset: random_seed: 42'
        rpm_profile=rpm_profile
    )

    # 5. Package Output
    # (Using raw physics output, skipping extra temperature model for pure physics validation)
    out = pd.DataFrame({
        'time': t,
        'imu_ax': sim_out['f_meas'][:, 0],
        'imu_ay': sim_out['f_meas'][:, 1],
        'imu_az': sim_out['f_meas'][:, 2],
        'imu_gx': sim_out['w_meas'][:, 0],
        'imu_gy': sim_out['w_meas'][:, 1],
        'imu_gz': sim_out['w_meas'][:, 2],
    })
    out['t_norm'] = out['time'] - out['time'].iloc[0]
    
    return out

def apply_temperature_model(
    time: np.ndarray,
    acc_in: np.ndarray,
    gyro_in: np.ndarray,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the 'ASM330LHH_Model_Upgraded' logic with TUNED noise levels.
    """
    dt = np.mean(np.diff(time))
    N = len(time)
    rng = np.random.default_rng(seed)
    
    # ------------------------------------------------------------------
    # A. SIMULATE TEMPERATURE PROFILE
    # ------------------------------------------------------------------
    T_ref = 25.0
    T_max = 55.0
    tau_temp = 600.0 
    temp_c = T_ref + (T_max - T_ref) * (1.0 - np.exp(-(time - time[0]) / tau_temp))
    dT = temp_c - T_ref

    # ------------------------------------------------------------------
    # B. CONFIGURATION (TUNED)
    # ------------------------------------------------------------------
    # 1. Deterministic Temp Coefficients
    acc_bias_slope   = np.array([0.00196, 0.00196, 0.00196]) 
    gyro_bias_slope  = np.array([0.005, 0.005, 0.005])       
    
    acc_scale_drift  = np.array([1e-4, 1e-4, 1e-4]) 
    gyro_scale_drift = np.array([1e-4, 1e-4, 1e-4])

    # 2. Stochastic Noise Params (TUNED HERE)
    # Bandwidth assumption: fs/2. 
    bw = 1.0 / (2.0 * dt)
    
    # --- TUNING CHANGES ---
    # Accel: INCREASED ~4x to match the "fuzz" of the real black line
    # Was: 60e-6 * 9.81
    acc_nd  = 250e-6 * 9.81  # ~2.5 mg/rtHz (matches visual "thickness" of real data)

    # Gyro: DECREASED ~8x because the red plot was way too fuzzy
    # Was: 0.0038
    gyro_nd = 0.0005         # 0.0005 dps/rtHz (Much smoother)
    # ----------------------

    acc_wn_std  = acc_nd * np.sqrt(bw)
    gyro_wn_std = gyro_nd * np.sqrt(bw)

    # Bias Instability (Gauss-Markov)
    tau_bias = 100.0
    acc_bias_inst  = np.array([20e-6*9.81]*3) 
    gyro_bias_inst = np.array([3.0/3600.0]*3) 
    
    # ------------------------------------------------------------------
    # C. APPLY DETERMINISTIC EFFECTS (Temp Bias & Scale)
    # ------------------------------------------------------------------
    s_acc  = 1.0 + np.outer(dT, acc_scale_drift)
    s_gyro = 1.0 + np.outer(dT, gyro_scale_drift)
    
    acc_out  = acc_in * s_acc
    gyro_out = gyro_in * s_gyro

    acc_out  += np.outer(dT, acc_bias_slope)
    gyro_out += np.outer(dT, gyro_bias_slope)

    # ------------------------------------------------------------------
    # D. APPLY STOCHASTIC EFFECTS (Bias Instability & White Noise)
    # ------------------------------------------------------------------
    # 1. Generate White Noise
    wn_acc  = rng.normal(0.0, acc_wn_std, size=(N, 3))
    wn_gyro = rng.normal(0.0, gyro_wn_std, size=(N, 3))
    
    # 2. Generate Bias Instability (Gauss-Markov)
    beta = np.exp(-dt / tau_bias)
    drive_acc  = acc_bias_inst * np.sqrt(1.0 - beta**2)
    drive_gyro = gyro_bias_inst * np.sqrt(1.0 - beta**2)

    w_acc_gm  = rng.normal(0.0, drive_acc, size=(N, 3))
    w_gyro_gm = rng.normal(0.0, drive_gyro, size=(N, 3))
    
    b_acc_gm  = np.zeros_like(acc_in)
    b_gyro_gm = np.zeros_like(gyro_in)
    
    curr_a = np.zeros(3)
    curr_g = np.zeros(3)
    for i in range(N):
        curr_a = beta * curr_a + w_acc_gm[i]
        curr_g = beta * curr_g + w_gyro_gm[i]
        b_acc_gm[i]  = curr_a
        b_gyro_gm[i] = curr_g

    # Sum it all up
    acc_out  += b_acc_gm + wn_acc
    gyro_out += b_gyro_gm + wn_gyro

    return acc_out, gyro_out, temp_c
'''
OLD VERSION, SHORTER PHYSICS NOT INVOKING FULL PIPELINE
def synth_with_motor_bands(df: pd.DataFrame) -> pd.DataFrame:
    t = df['time'].to_numpy()
    
    # Logic Update: Use input directly. 
    # Isaac Sim 'imu_ax' is already Specific Force (Gravity included).
    f_B_true = df[['acc_x','acc_y','acc_z']].to_numpy()
    w_B = df[['ang_vel_x','ang_vel_y','ang_vel_z']].to_numpy()

    # --- INCREASED NOISE PARAMETERS ---
    
    # Franka-like motor harmonic parameters
    # Gains have been doubled to increase spike amplitude
    modes_acc = [
        ResonantMode(f0=75.0,  zeta=0.02, gain=0.10, axes=(0,1,2)), # Was 0.05
        ResonantMode(f0=95.0,  zeta=0.03, gain=0.04, axes=(0,1,2)), # Was 0.02
        ResonantMode(f0=120.0, zeta=0.03, gain=0.08, axes=(2,)),    # Was 0.04
        ResonantMode(f0=130.0, zeta=0.03, gain=0.06, axes=(2,)),    # Was 0.03
    ]
    modes_gyr = [
        ResonantMode(f0=6.5,   zeta=0.07, gain=0.04, axes=(0,1,2)), # Was 0.02
        ResonantMode(f0=8.0,   zeta=0.08, gain=0.03, axes=(0,1,2)), # Was 0.015
        ResonantMode(f0=75.0,  zeta=0.03, gain=0.008, axes=(2,)),   # Was 0.004
        ResonantMode(f0=120.0, zeta=0.03, gain=0.008, axes=(2,)),   # Was 0.004
    ]

    # Convert 75 Hz to RPM for the motor base frequency in the excitation model.
    rpm = np.full_like(t, 75.0*60.0)  # 4500 rpm constant

    

    # White noise STD increased (approx 2x)
    w_meas, a_meas, _info = simulate_imu_with_vibration(
        t, true_w_B=w_B, true_a_B=f_B_true, rpm_profile=rpm,
        modes_accel=modes_acc, modes_gyro=modes_gyr,
        accel_white_std=0.04,   # Was 0.02
        gyro_white_std=0.005,   # Was 0.0025
        gyro_bias_rw_std=2e-5, g_sensitivity=0.002,
        quantize_accel_mg=None, quantize_gyro_dps=None, seed=123)

    out = pd.DataFrame({
        'time': t,
        'imu_ax': a_meas[:,0],
        'imu_ay': a_meas[:,1],
        'imu_az': a_meas[:,2],
        'imu_gx': w_meas[:,0],
        'imu_gy': w_meas[:,1],
        'imu_gz': w_meas[:,2],
    })
    out['t_norm'] = out['time'] - out['time'].iloc[0]
    return out
'''

'''
# -----------------------------------------------------------------------------
# BACKUP: Original function without Temperature/Sensor Model, but full pipeline
# -----------------------------------------------------------------------------
def synth_with_motor_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesizes IMU data using the sophisticated 'synth_measurements' physics engine
    by adapting the flat CSV data into the required kinematic tensors.
    """
    # 1. Extract Time and Calculate dt
    t = df['time'].to_numpy()
    dt = np.mean(np.diff(t))
    N = len(t)

    # 2. Extract Measurements (Input is Specific Force and Gyro)
    # Isaac Sim 'imu_ax' is Specific Force (f_B) which includes gravity.
    f_B_input = df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    w_B_input = df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].to_numpy()

    # 3. Create Mock Kinematics
    # To pass "raw" data through the physics engine without rotating it, 
    # we assume the Body frame is aligned with the World frame (Identity Rotation).
    R_BW = np.repeat(np.eye(3)[np.newaxis, :, :], N, axis=0) # Body-to-World
    
    # The simulator computes: f_meas = R_BW.T @ (a_W - g_W)
    # We want f_meas ≈ f_B_input.
    # Since R_BW is Identity, we need: f_B_input = a_W - g_W
    # Therefore: a_W = f_B_input + g_W
    g_W = np.array([0.0, 0.0, 9.80665])
    a_W = f_B_input + g_W  # "World" acceleration that produces the observed specific force
    
    # Calculate angular acceleration (needed for lever arm effects)
    wdot_B = np.gradient(w_B_input, dt, axis=0)
    
    # Lever arm (Assume 0 since we want to modify the signal "in place")
    r_lever = np.zeros(3)

    # 4. Construct Configuration Dictionary
    # We map the hardcoded variables from your old script into the YAML structure.
    
    # Convert Standard Deviation (old script) to Noise Density (new script expectation)
    # Formula: density = sigma * sqrt(dt)
    acc_sigma = 0.04   # Was 0.04 in your old script
    gyr_sigma = 0.005  # Was 0.005 in your old script
    acc_density = acc_sigma * np.sqrt(dt)
    gyr_density = gyr_sigma * np.sqrt(dt)

    # Define Vibration Modes (Mapped from your ResonantMode list)
    # Note: Gains doubled as per your previous logic
    accel_modes_cfg = [
        {'f0': 75.0,  'zeta': 0.02, 'gain': 0.10, 'axes': [0, 1, 2]},
        {'f0': 95.0,  'zeta': 0.03, 'gain': 0.04, 'axes': [0, 1, 2]},
        {'f0': 120.0, 'zeta': 0.03, 'gain': 0.08, 'axes': [2]},
        {'f0': 130.0, 'zeta': 0.03, 'gain': 0.06, 'axes': [2]},
    ]
    gyro_modes_cfg = [
        {'f0': 6.5,   'zeta': 0.07, 'gain': 0.04,  'axes': [0, 1, 2]},
        {'f0': 8.0,   'zeta': 0.08, 'gain': 0.03,  'axes': [0, 1, 2]},
        {'f0': 75.0,  'zeta': 0.03, 'gain': 0.008, 'axes': [2]},
        {'f0': 120.0, 'zeta': 0.03, 'gain': 0.008, 'axes': [2]},
    ]

    cfg = {
        'imu': {
            'quantization_bits': 16, # Matches YAML
            'accel_fs_g': 8.0,       # Matches YAML
            'gyro_fs_dps': 2000.0,   # Matches YAML
            'misalignment_pct': [0.0, 0.0], # Disable spatial misalignment to keep axes clean
            'accel': {
                'scale_ppm': [0, 0],
                'bias_init': [0.0, 0.0],
                'bias_tau_s': [3600, 3600], # Slow drift
                'noise_density': [acc_density, acc_density] 
            },
            'gyro': {
                'scale_ppm': [0, 0],
                'bias_init': [0.0, 0.0],
                'bias_tau_s': [3600, 3600],
                'noise_density': [gyr_density, gyr_density]
            }
        },
        'vibration': {
            'g_sensitivity': 0.002,
            # Fundamental motor harmonics (1.0 = 4500 RPM)
            'motor_harmonics': {1: 1.0, 2: 0.35, 3: 0.2}, 
            # Broadband noise floor is needed to excite modes that don't align with motor harmonics
            'floor_noise_sigma': 0.4, 
            'floor_noise_ar': 0.96,
            'floor_noise_ma': 0.2,
            'accel_modes': accel_modes_cfg,
            'gyro_modes': gyro_modes_cfg
        }
    }

    # 5. RPM Profile (4500 RPM constant)
    rpm_profile = np.full_like(t, 75.0 * 60.0)

    # 6. Run Simulation
    # Note: We pass R_BW (Identity) and calculated a_W
    sim_out, _params = synth_measurements(
        R_WB=R_BW, # Function arg is named R_WB, but expects Body-to-World logic for projection? 
                   # Actually, standard naming R_WB usually means World-to-Body. 
                   # However, since we use Identity, R_WB = R_BW = I. 
        w_B=w_B_input,
        a_W=a_W,
        wdot_B=wdot_B,
        r_lever=r_lever,
        cfg=cfg,
        dt=dt,
        seed=123,
        rpm_profile=rpm_profile
    )

    # 7. Package Output
    out = pd.DataFrame({
        'time': t,
        'imu_ax': sim_out['f_meas'][:, 0],
        'imu_ay': sim_out['f_meas'][:, 1],
        'imu_az': sim_out['f_meas'][:, 2],
        'imu_gx': sim_out['w_meas'][:, 0],
        'imu_gy': sim_out['w_meas'][:, 1],
        'imu_gz': sim_out['w_meas'][:, 2],
    })
    out['t_norm'] = out['time'] - out['time'].iloc[0]
    
    return out
'''

def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    
    x = x - np.mean(x); y = y - np.mean(y)
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0:
        return float('nan')
    return float(np.corrcoef(x/sx, y/sy)[0,1])


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Inject motor-like vibration into clean logs and compare to real IMU')
    ap.add_argument('--logs_root', default='logs_test')
    ap.add_argument('--out', default='runs/vib_compare')
    ap.add_argument('--crop_start_s', type=float, default=0.0) # Changed default to 0.0 to avoid shifting issues
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    traj_dirs = sorted(d for d in glob.glob(os.path.join(args.logs_root, 'traj_*')) if os.path.isdir(d))
    summary_rows: List[List[str]] = [['traj','corr_ax','corr_ay','corr_az','corr_gx','corr_gy','corr_gz']]

    for traj in traj_dirs:
        # Search for files
        clean_glob = sorted(glob.glob(os.path.join(traj, 'imu_400hz_*.csv')))
        real_glob  = sorted(glob.glob(os.path.join(traj, 'real_400hz_*.csv')))
        
        if not clean_glob or not real_glob:
            print(f'Skipping {traj} (missing imu_400hz or real_400hz)')
            continue
            
        clean_path = clean_glob[0]
        real_path  = real_glob[0]
        base = os.path.basename(traj) 
        
        # EXTRACT ID
        traj_id = base.split('_')[-1] 

        # Load and synthesize
        clean_df = load_clean(clean_path)
        synth_df = synth_with_motor_bands(clean_df)

        # SAVE
        sim_out_filename = f'sim_noisy_{traj_id}.csv'
        sim_out_path = os.path.join(traj, sim_out_filename)
        synth_df.to_csv(sim_out_path, index=False)
        print(f"[{base}] Generated: {sim_out_path}")

        # Load real 
        real_df = pd.read_csv(real_path)
        real_df.columns = real_df.columns.str.strip()
        real_df['t_norm'] = real_df['time'] - real_df['time'].iloc[0]

        # Optional crop
        if args.crop_start_s > 0:
            synth_df = synth_df[synth_df['t_norm'] >= args.crop_start_s].reset_index(drop=True)
            real_df  = real_df[ real_df['t_norm']  >= args.crop_start_s].reset_index(drop=True)

        # ---------------------------------------------------------
        # FIX: DISABLE AUTO-ALIGNMENT
        # We verified visually that axes are already correct. 
        # Running the solver risks scrambling them.
        # ---------------------------------------------------------
        print(f"[{base}] Skipping auto-alignment (trusting manual mapping)...")
        
        # Force Identity Matrices (No rotation/permutation)
        A_acc = np.eye(3)
        A_gyr = np.eye(3)
        
        # Just copy the dataframe, do not permute
        synth_df_al = synth_df.copy()
        # ---------------------------------------------------------

        # Plot overlay
        out_path = os.path.join(args.out, f'{base}_vib_vs_real.png')
        
        # NOTE: Ensure plot_pair handles the columns correctly
        # The synth_df has 'imu_ax', real_df has 'acc_x_g'
        plot_pair(synth_df_al, real_df, f'{base}: vib-synth vs real', out_path)
        print('Wrote', out_path)

        # Correlations 
        t_ref = real_df['t_norm'].to_numpy()
        
        metrics: Dict[str, float] = {}
        for s_col, r_col, name in [
            ('imu_ax','acc_x_g','ax'),
            ('imu_ay','acc_y_g','ay'),
            ('imu_az','acc_z_g','az'),
            ('imu_gx','gyro_x_rad_s','gx'),
            ('imu_gy','gyro_y_rad_s','gy'),
            ('imu_gz','gyro_z_rad_s','gz'),
        ]:
            s_al = np.interp(t_ref, synth_df_al['t_norm'].to_numpy(), synth_df_al[s_col].to_numpy())
            r = real_df[r_col].to_numpy()
            metrics[f'corr_{name}'] = corrcoef(s_al, r)

        summary_rows.append([
            base,
            *(f"{metrics[k]:.6f}" for k in ['corr_ax','corr_ay','corr_az','corr_gx','corr_gy','corr_gz'])
        ])

    # Save summary
    out_csv = os.path.join(args.out, 'correlations.csv')
    with open(out_csv, 'w') as f:
        for row in summary_rows:
            f.write(','.join(row) + '\n')
    print('Wrote', out_csv)

if __name__ == '__main__':
    main()