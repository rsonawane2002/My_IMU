import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
import math

# --- 1. Define Model Architecture (Must match training exactly) ---
# (Copy-paste your class definitions here or import them if in a module)
# For this script to be standalone, I will assume the classes are available.
# Ensure MambaSeq2Seq and PhysicsMambaBlock are defined above or imported.

from mamba_sim2real_spectral import MambaSeq2Seq, CONFIG # Adjust import as needed

def load_inference_model(model_path, scaler_path, device):
    print(f"Loading model from {model_path}...")
   
    # Initialize Model Structure
    model = MambaSeq2Seq(
        input_dim=CONFIG['input_dim'],
        output_dim=CONFIG['output_dim'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand']
    ).to(device)
   
    # Load Weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
   
    # Load Scaler
    scaler = joblib.load(scaler_path)
   
    return model, scaler

'''
def process_single_file(sim_csv_path, scaler):
    """
    Prepares a single CSV file for inference, matching the training preprocessing.
    """
    df = pd.read_csv(sim_csv_path)
   
    # 1. Feature Engineering (Same as Dataset)
    # Calculate Joint Velocities
    joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
    imu_cols = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
   
    # Calculate dt (Time step)
    time_vals = df['time'].values
    dt_vals = np.diff(time_vals, prepend=time_vals[0])
    # Handle first element 0 dt
    dt_vals[0] = np.mean(dt_vals[1:])
   
    # Calc velocities
    vels = df[joint_cols].diff().fillna(0).values / dt_vals[:, None]
   
    # Stack features: [IMU (6) | Joints (7) | Velocities (7)]
    raw_features = np.hstack([
        df[imu_cols].values,
        df[joint_cols].values,
        vels
    ])
   
    # 2. Scaling
    # We only scale the features, not the dt input
    scaled_features = scaler.transform(raw_features)
   
    # 3. Create Tensor
    # Model expects: [Batch, Seq, Features + dt]
    # We create a batch of size 1
    dt_col = dt_vals.reshape(-1, 1)
    input_np = np.hstack([scaled_features, dt_col])
   
    input_tensor = torch.FloatTensor(input_np).unsqueeze(0) # Add batch dim
   
    return input_tensor, df[imu_cols].values # Return original sim IMU for reconstruction
'''
def process_single_file(sim_csv_path, scaler):
    df = pd.read_csv(sim_csv_path)
    joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
    imu_cols = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
   
    time_vals = df['time'].values
    # MATCH TRAINING: Use mean DT for the whole trajectory
    dt_mean = np.mean(np.diff(time_vals))
    if dt_mean <= 0: dt_mean = 0.01667 # Safety fallback for 60Hz
   
    # Calculate velocities using the stable mean DT
    vels = df[joint_cols].diff().fillna(0).values / dt_mean
   
    raw_features = np.hstack([
        df[imu_cols].values,
        df[joint_cols].values,
        vels
    ])
   
    scaled_features = scaler.transform(raw_features)
   
    # Create the DT column using the same stable mean DT
    dt_col = np.full((len(df), 1), dt_mean)
    input_np = np.hstack([scaled_features, dt_col])
    input_tensor = torch.FloatTensor(input_np).unsqueeze(0)
   
    return input_tensor, df[imu_cols].values

def generate_sim2real(model, input_tensor, original_sim_imu, device, num_samples=1):
    """
    Runs inference and generates probabilistic outputs.
    """
    input_tensor = input_tensor.to(device)
   
    with torch.no_grad():
        # 1. Forward Pass
        mu, log_var = model(input_tensor)
       
        # 2. Extract Mean and Sigma
        mu = mu.squeeze(0).cpu().numpy()
        sigma = torch.exp(0.5 * log_var).squeeze(0).cpu().numpy()
       
    # 3. Sampling Loop
    # We can generate multiple "possible realities" if needed, usually just 1 is fine.
    generated_outputs = []
   
    for _ in range(num_samples):
        # Generate random white noise
        epsilon = np.random.normal(0, 1, size=mu.shape)
       
        # The Core Equation: Predicted Residual = Mean + (Sigma * Noise)
        predicted_residual = mu + (sigma * epsilon)
       
        # Add Residual to Original Sim Data to get "Sim2Real"
        # Real = Sim + Residual
        sim2real_imu = original_sim_imu + predicted_residual
        generated_outputs.append(sim2real_imu)
       
    return generated_outputs[0], mu, sigma # Return first sample, plus physics stats

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Paths
    MODEL_PATH = "best_model_probabilistic.pth"
    SCALER_PATH = "scaler.pkl"
    TEST_FILE = "./logs_2/traj_270/imu_270.csv" # Change this
    OUTPUT_FILE = "sim2real_output.csv"
   
    # 1. Load
    model, scaler = load_inference_model(MODEL_PATH, SCALER_PATH, device)
   
    # 2. Prepare Data
    print(f"Processing {TEST_FILE}...")
    input_tensor, original_sim_imu = process_single_file(TEST_FILE, scaler)
   
    # 3. Run Inference
    print("Generating Sim-to-Real data...")
    final_fake_real, bias_correction, noise_envelope = generate_sim2real(
        model, input_tensor, original_sim_imu, device
    )
   
    # 4. Save Results
    # Create a DataFrame combining everything for analysis
    cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
   
    df_out = pd.DataFrame(final_fake_real, columns=[f"{c}_sim2real" for c in cols])
   
    # Add original sim for comparison
    df_orig = pd.DataFrame(original_sim_imu, columns=[f"{c}_original" for c in cols])
   
    # Add the "Physics Correction" (Mean) alone - useful for debugging
    df_bias = pd.DataFrame(bias_correction, columns=[f"{c}_bias" for c in cols])
   
    # Add the "Noise Level" (Sigma) - useful to see where model thinks it's noisy
    df_sigma = pd.DataFrame(noise_envelope, columns=[f"{c}_sigma" for c in cols])
   
    final_df = pd.concat([df_orig, df_out, df_bias, df_sigma], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False)
   
    print(f"Done! Saved generated IMU data to {OUTPUT_FILE}")
    print("Columns explanation:")
    print(" - _original: The clean simulation input")
    print(" - _sim2real: The final noisy output (Use this for your Kalman Filter/RL)")
    print(" - _bias: The deterministic correction (Center line)")
    print(" - _sigma: The predicted standard deviation of noise")