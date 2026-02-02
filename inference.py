import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import math
import os
import sys  # <--- ADDED
# from scipy.signal import savgol_filter  <-- REMOVED
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat

# ==============================================================================
# 1. MODEL DEFINITIONS (Must match training exactly)
# ==============================================================================

# Configuration used for initialization
CONFIG = {
    'input_dim': 6,
    'output_dim': 6,
    'd_model': 256,
    'd_state': 64,
    'd_conv': 4,
    'expand': 2
}

class PhysicsMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
       
        A = repeat(np.arange(1, self.d_state + 1, dtype=np.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.tensor(A)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, dt_vals):
        batch, seq, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq]
        x = self.activation(x)
       
        x_for_proj = x.transpose(1, 2)
        x_dbl = self.x_proj(x_for_proj)
        _, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
       
        B = B.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()
        dt = self.dt_proj(dt_vals).transpose(1, 2).contiguous()
        z = z.transpose(1, 2).contiguous()

        y = selective_scan_fn(
            x, dt, A=-torch.exp(self.A_log.float()), B=B, C=C, D=self.D.float(),
            z=z, delta_bias=None, delta_softplus=True, return_last_state=False,
        )
        return self.out_proj(y.transpose(1, 2))


class MambaSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_state, d_conv, expand):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.mamba_block = PhysicsMambaBlock(d_model, d_state, d_conv, expand)
        self.output_projection = nn.Linear(d_model, output_dim * 2)
   
    def forward(self, x_full):
        x_features = x_full[:, :, :-1]
        dt_vals = x_full[:, :, -1:]    
        x_emb = self.input_projection(x_features)
        mamba_out = self.mamba_block(x_emb, dt_vals)
        raw_output = self.output_projection(mamba_out)
        mu, log_var = raw_output.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10, max=5)
        return mu, log_var

# ==============================================================================
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
    df = pd.read_csv(sim_csv_path)
    imu_cols = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
    time_vals = df['time'].values
    dt_vals = np.diff(time_vals, prepend=time_vals[0])
    dt_vals[0] = np.mean(dt_vals[1:])
    raw_features = df[imu_cols].values
    scaled_features = scaler.transform(raw_features)
    dt_col = dt_vals.reshape(-1, 1)
    input_np = np.hstack([scaled_features, dt_col])
    input_tensor = torch.FloatTensor(input_np).unsqueeze(0)
    return input_tensor, df[imu_cols].values

def generate_sim2real(model, input_tensor, original_sim_imu, device, num_samples=1):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        mu, log_var = model(input_tensor)
        mu = mu.squeeze(0).cpu().numpy()
        sigma = torch.exp(0.5 * log_var).squeeze(0).cpu().numpy()
    generated_outputs = []
    for _ in range(num_samples):
        epsilon = np.random.normal(0, 1, size=mu.shape)
        predicted_residual = mu + (sigma * epsilon)
        sim2real_imu = original_sim_imu + predicted_residual
        generated_outputs.append(sim2real_imu)
    return generated_outputs[0], mu, sigma

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- CHANGED: Argument Parsing ---
    if len(sys.argv) < 2:
        print("Usage: python inference.py <traj_id>")
        sys.exit(1)
    
    traj_id = sys.argv[1]
    # ---------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Paths (Adjust these)
    MODEL_PATH = "best_model_probabilistic.pth"
    SCALER_PATH = "scaler.pkl"
    TEST_FILE = f"./logs_2/traj_{traj_id}/imu_400hz_{traj_id}.csv"  # --- CHANGED: Uses traj_id
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