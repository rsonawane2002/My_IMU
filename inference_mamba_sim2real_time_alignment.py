import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import math
import os
from scipy.signal import savgol_filter  # <--- Essential for matching training inputs
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
from mamba_sim2real_spectral_loss_time_alignment import PhysicsMambaBlock, MambaSeq2Seq


# ==============================================================================
# 1. MODEL DEFINITIONS (Must match training exactly)
# ==============================================================================

# Configuration used for initialization
# (Ideally load this from a saved config file, but hardcoded here for simplicity)
CONFIG = {
    'input_dim': 20,
    'output_dim': 6,
    'd_model': 256,
    'd_state': 64,
    'd_conv': 4,
    'expand': 2
}

# 2. INFERENCE HELPERS
# ==============================================================================

def load_inference_model(model_path, scaler_path, device):
    print(f"Loading model from {model_path}...")
   
    model = MambaSeq2Seq(
        input_dim=CONFIG['input_dim'],
        output_dim=CONFIG['output_dim'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand']
    ).to(device)
   
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
   
    scaler = joblib.load(scaler_path)
    return model, scaler

def process_single_file(sim_csv_path, scaler):
    """
    Prepares a single CSV file for inference, matching the training preprocessing.
    """
    df = pd.read_csv(sim_csv_path)
   
    # 1. Feature Engineering
    joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
    imu_cols = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
   
    # Calculate dt
    time_vals = df['time'].values
    dt_vals = np.diff(time_vals, prepend=time_vals[0])
    dt_vals[0] = np.mean(dt_vals[1:])
   
    # --- UPDATED: Savitzky-Golay Smoothing ---
    # Matches the training script to prevent noisy input artifacts
    dt_sim = np.mean(dt_vals)
    if dt_sim <= 0: dt_sim = 0.016

    vels = savgol_filter(
        df[joint_cols].values,
        window_length=7,   # Must match training
        polyorder=3,       # Must match training
        deriv=1,
        delta=dt_sim,
        axis=0
    )
    # -----------------------------------------
   
    # Stack features: [IMU (6) | Joints (7) | Velocities (7)]
    raw_features = np.hstack([
        df[imu_cols].values,
        df[joint_cols].values,
        vels
    ])
   
    # 2. Scaling
    scaled_features = scaler.transform(raw_features)
   
    # 3. Create Tensor [Batch=1, Seq, Feat+dt]
    dt_col = dt_vals.reshape(-1, 1)
    input_np = np.hstack([scaled_features, dt_col])
   
    input_tensor = torch.FloatTensor(input_np).unsqueeze(0)
   
    return input_tensor, df[imu_cols].values

def generate_sim2real(model, input_tensor, original_sim_imu, device, num_samples=1):
    """
    Runs inference and generates probabilistic outputs.
    """
    input_tensor = input_tensor.to(device)
   
    with torch.no_grad():
        mu, log_var = model(input_tensor)
       
        mu = mu.squeeze(0).cpu().numpy()
        sigma = torch.exp(0.5 * log_var).squeeze(0).cpu().numpy()
       
    generated_outputs = []
   
    for _ in range(num_samples):
        # Generate random white noise
        epsilon = np.random.normal(0, 1, size=mu.shape)
       
        # Predicted Residual = Mean + (Sigma * Noise)
        predicted_residual = mu + (sigma * epsilon)
       
        # Real = Sim + Residual
        sim2real_imu = original_sim_imu + predicted_residual
        generated_outputs.append(sim2real_imu)
       
    return generated_outputs[0], mu, sigma

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Paths (Adjust these)
    MODEL_PATH = "best_model_probabilistic.pth"
    SCALER_PATH = "scaler.pkl"
    TEST_FILE = "./logs_2/traj_350/imu_400hz_350.csv"
    OUTPUT_FILE = "sim2real_output.csv"
   
    # 1. Load
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler file not found.")
        exit()

    model, scaler = load_inference_model(MODEL_PATH, SCALER_PATH, device)
   
    # 2. Prepare Data
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file {TEST_FILE} not found.")
        exit()
       
    print(f"Processing {TEST_FILE}...")
    input_tensor, original_sim_imu = process_single_file(TEST_FILE, scaler)
   
    # 3. Run Inference
    print("Generating Sim-to-Real data...")
    final_fake_real, bias_correction, noise_envelope = generate_sim2real(
        model, input_tensor, original_sim_imu, device
    )
   
    # 4. Save Results
    cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
   
    df_out = pd.DataFrame(final_fake_real, columns=[f"{c}_sim2real" for c in cols])
    df_orig = pd.DataFrame(original_sim_imu, columns=[f"{c}_original" for c in cols])
    df_bias = pd.DataFrame(bias_correction, columns=[f"{c}_bias" for c in cols])
    df_sigma = pd.DataFrame(noise_envelope, columns=[f"{c}_sigma" for c in cols])
   
    final_df = pd.concat([df_orig, df_out, df_bias, df_sigma], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False)
   
    print(f"Done! Saved generated IMU data to {OUTPUT_FILE}")