import os
import glob
import itertools
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- PART 1: MATH HELPERS (GRAVITY & ROTATION) ---

def quaternion_to_rotation_matrix(q):
    """
    Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_all_signed_permutations():
    """
    Generates all 24 matrices representing 90-degree rotations 
    (axis swaps and sign flips) for alignment search.
    """
    matrices = []
    for p in itertools.permutations([0, 1, 2]):
        for s in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3))
            R[0, p[0]] = s[0]
            R[1, p[1]] = s[1]
            R[2, p[2]] = s[2]
            matrices.append(R)
    return matrices

ROTATIONS = get_all_signed_permutations()

def get_best_alignment(sim_gyro, real_gyro):
    """
    Finds the rotation matrix R that maximizes correlation between sim_gyro and real_gyro.
    """
    best_R = np.eye(3)
    best_score = -np.inf
    
    if len(real_gyro) < 10: 
        return np.eye(3), 0.0

    # Center data for correlation
    r_centered = real_gyro - np.mean(real_gyro, axis=0)
    
    for R in ROTATIONS:
        # Apply rotation: Sim_Rot = Sim @ R.T
        s_rot = sim_gyro @ R.T
        s_centered = s_rot - np.mean(s_rot, axis=0)
        
        # Fast correlation score (Sum of dot products / Normalization)
        numerator = np.sum(r_centered * s_centered)
        denom = np.sqrt(np.sum(r_centered**2) * np.sum(s_centered**2))
        
        score = 0.0 if denom == 0 else numerator / denom
        
        if score > best_score:
            best_score = score
            best_R = R
            
    return best_R, best_score

# --- PART 2: MAIN PROCESSING LOGIC ---

def process_log(clean_csv_path):
    base_name = os.path.basename(clean_csv_path)
    dir_name = os.path.dirname(clean_csv_path)
    
    # 1. Load Clean Data
    try:
        df_sim = pd.read_csv(clean_csv_path)
        df_sim.columns = df_sim.columns.str.strip()
    except Exception as e:
        print(f"[{base_name}] Error reading: {e}")
        return

    # Check required columns
    req_cols = ['acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
    if not all(c in df_sim.columns for c in req_cols):
        print(f"[{base_name}] Skipping: Missing required kinematic columns.")
        return

    # 2. Apply Gravity Correction (Physics Step)
    # f_B = a_kinematic - g_B
    a_B_input = df_sim[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    quats = df_sim[['quat_w', 'quat_x', 'quat_y', 'quat_z']].to_numpy()
    
    # Convert all quaternions to rotation matrices
    R_WB = np.array([quaternion_to_rotation_matrix(q) for q in quats])
    g_W = np.array([0.0, 0.0, 9.81])
    
    # Rotate gravity into Body frame: g_B = R_WB^T * g_W
    # Einsum magic: 'nij,j->ni' does matrix-vector mult for each row n
    g_B = np.einsum('nij,j->ni', R_WB, g_W)
    
    # Calculate Specific Force (this is what the accelerometer actually measures)
    sim_acc_with_gravity = a_B_input - g_B
    sim_gyro_raw = df_sim[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].to_numpy()

    # 3. Find Alignment (Comparison Step)
    # Look for matching Real log
    real_files = glob.glob(os.path.join(dir_name, 'real_imu_*.csv'))
    
    R_align = np.eye(3) # Default Identity
    align_score = 0.0
    alignment_found = False

    if real_files:
        real_csv_path = real_files[0]
        try:
            df_real = pd.read_csv(real_csv_path)
            df_real.columns = df_real.columns.str.strip()
            
            # Identify Time and Gyro columns
            t_real_col = next((c for c in ['time_sec', 'time', 't'] if c in df_real.columns), None)
            
            # Flexible gyro column search
            gx_col = next((c for c in ['gyro_x_rad_s', 'gyro_x'] if c in df_real.columns), None)
            gy_col = next((c for c in ['gyro_y_rad_s', 'gyro_y'] if c in df_real.columns), None)
            gz_col = next((c for c in ['gyro_z_rad_s', 'gyro_z'] if c in df_real.columns), None)

            if t_real_col and gx_col:
                # Prepare data for interpolation
                t_sim = df_sim['time'].to_numpy(); t_sim -= t_sim[0]
                t_real = df_real[t_real_col].to_numpy(); t_real -= t_real[0]

                # Crop overlap
                t_max = min(t_sim[-1], t_real[-1])
                if t_max > 0.1:
                    mask = t_real <= t_max
                    t_real = t_real[mask]
                    real_gyro = df_real[[gx_col, gy_col, gz_col]].to_numpy()[mask]

                    # Interpolate Sim Gyro to Real Time
                    f_interp = interp1d(t_sim, sim_gyro_raw, axis=0, fill_value="extrapolate")
                    sim_gyro_interp = f_interp(t_real)

                    # Solve Alignment
                    R_align, align_score = get_best_alignment(sim_gyro_interp, real_gyro)
                    alignment_found = True
        except Exception as e:
            print(f"[{base_name}] Warning: Alignment failed ({e}). Using Identity.")

    # 4. Apply Alignment Rotation
    # We rotate both the gravity-corrected Accel and the Gyro
    # New = Old @ R.T
    final_acc = sim_acc_with_gravity @ R_align.T
    final_gyro = sim_gyro_raw @ R_align.T

    # 5. Save Output
    # Create output dataframe
    df_out = df_sim.copy()
    
    # Update columns with calculated & aligned values
    df_out['acc_x'] = final_acc[:, 0]
    df_out['acc_y'] = final_acc[:, 1]
    df_out['acc_z'] = final_acc[:, 2]
    
    df_out['ang_vel_x'] = final_gyro[:, 0]
    df_out['ang_vel_y'] = final_gyro[:, 1]
    df_out['ang_vel_z'] = final_gyro[:, 2]

    # Generate filename: clean_X.csv -> clean_X_with_gravity.csv
    name_part = os.path.splitext(base_name)[0]
    out_path = os.path.join(dir_name, f"{name_part}_with_gravity.csv")
    
    df_out.to_csv(out_path, index=False)
    
    status = f"Aligned (Score: {align_score:.2f})" if alignment_found else "Gravity Added (No Real Log)"
    print(f"[{base_name}] -> {os.path.basename(out_path)} | {status}")

def main():
    logs_root = 'logs'
    # Pattern to find only the original clean CSVs
    files = sorted(glob.glob(os.path.join(logs_root, 'traj_*', 'clean_*.csv')))
    
    print(f"Found {len(files)} files. Starting processing...")
    print("-" * 60)

    for csv_path in files:
        fname = os.path.basename(csv_path)
        
        # STRICT FILTER: Skip files we generated or intermediate files
        if '_with_gravity' in fname: continue
        if '_processed' in fname: continue
        if '_aligned' in fname: continue
        
        process_log(csv_path)

if __name__ == '__main__':
    main()