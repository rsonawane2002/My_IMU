import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
from scipy.interpolate import interp1d
import joblib
import math
import os

# ================= CONFIGURATION =================
# Point this to a test file you want to visualize
TEST_SIM_CSV = "./logs_2/traj_260/imu_260.csv"
TEST_REAL_CSV = "./logs_2/traj_260/real_filtered_imu_260.csv"

# FIX 1: Match the filename saved in your training loop
MODEL_PATH = "best_model.pth" 
SCALER_PATH = "scaler.pkl"

# FIX 2: Match training config (20 inputs, not 24)
CONFIG = {
    'input_dim': 20, 
    'output_dim': 6, 
    'd_model': 256,
    'd_state': 64,
    'd_conv': 4,
    'expand': 2,
    'seq_len': 512
}
# =================================================

# --- 1. MODEL DEFINITION (MUST MATCH TRAINING EXACTLY) ---
class PhysicsMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False) 
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Init (Simplified for inference loading)
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        
        A = repeat(np.arange(1, d_state + 1, dtype=np.float32), "n -> d n", d=self.d_inner)
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
        
        x_dbl = self.x_proj(x.transpose(1, 2))
        _, B, C = torch.split(x_dbl, [self.dt_rank, 64, 64], dim=-1) # Hardcoded d_state=64 per config
        
        dt = self.dt_proj(dt_vals).transpose(1, 2)
        B = B.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()
        z = z.transpose(1, 2).contiguous()

        y = selective_scan_fn(x, dt, A=-torch.exp(self.A_log.float()), B=B, C=C, D=self.D.float(), z=z, delta_bias=None, delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))

class MambaSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_state, d_conv, expand):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.mamba_block = PhysicsMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x_full):
        return self.output_projection(self.mamba_block(self.input_projection(x_full[:,:,:-1]), x_full[:,:,-1:]))

def prepare_input(sim_path, real_path, scaler):
    sim_df = pd.read_csv(sim_path)
    real_df = pd.read_csv(real_path)

    # Time Alignment
    sim_time = sim_df['time'].values - sim_df['time'].values[0]
    real_time = real_df['time'].values - real_df['time'].values[0]

    # Create common 100Hz timeline
    max_t = min(sim_time[-1], real_time[-1])
    t_common = np.arange(0, max_t, 0.01) # 100 Hz

    # Helper to interpolate columns
    def resample(t_src, data_src, t_targ):
        f = interp1d(t_src, data_src, axis=0, fill_value="extrapolate")
        return f(t_targ)

    # 1. Prepare Sim Features (X)
    joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
    imu_cols = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz'] # Check column names match your CSV
    
    # Calculate velocities
    dt_sim = 0.01
    sim_vels = sim_df[joint_cols].diff().fillna(0).values / dt_sim
    
    # FIX 3: Remove Quaternions. Stack only what was trained (20 dims)
    sim_raw = np.hstack([
        sim_df[imu_cols].values,
        sim_df[joint_cols].values,
        sim_vels
    ])
    
    # Resample Sim to 100Hz
    sim_resampled = resample(sim_time, sim_raw, t_common)

    # 2. Prepare Real Targets (Y)
    real_raw = real_df[['acc_x_g', 'acc_y_g', 'acc_z_g', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
    real_resampled = resample(real_time, real_raw, t_common)

    # 3. Scale Input (Now valid because sim_resampled has 20 cols)
    feat_scaled = scaler.transform(sim_resampled)
    
    # 4. Add DT column
    dt_col = np.full((len(feat_scaled), 1), 0.01)
    x_input = np.hstack([feat_scaled, dt_col])

    return t_common, x_input, sim_resampled, real_resampled

# --- 3. MAIN INFERENCE ---
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # Load Components
    scaler = joblib.load(SCALER_PATH)
    model = MambaSeq2Seq(CONFIG['input_dim'], CONFIG['output_dim'], CONFIG['d_model'], CONFIG['d_state'], CONFIG['d_conv'], CONFIG['expand'])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Prepare Data
    print(f"Processing {TEST_SIM_CSV}...")
    t, x_np, sim_raw, real_raw = prepare_input(TEST_SIM_CSV, TEST_REAL_CSV, scaler)
    
    # Run Model
    x_tensor = torch.FloatTensor(x_np).unsqueeze(0).to(device) # Add batch dim
    
    with torch.no_grad():
        # Predict Residual
        residual_pred = model(x_tensor).squeeze(0).cpu().numpy()

    # Sim2Real = Sim + Predicted Residual
    # Note: sim_raw[:, :6] is the IMU part
    sim2real = sim_raw[:, :6] + residual_pred

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    labels = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

    for i in range(6):
        ax = axes[i]
        ax.plot(t, real_raw[:, i], label='Real (Ground Truth)', color='black', alpha=0.3, linewidth=2)
        ax.plot(t, sim_raw[:, i], label='Sim (Original)', color='orange', linestyle='--', alpha=0.8)
        ax.plot(t, sim2real[:, i], label='Sim2Real (Generated)', color='blue', alpha=0.7, linewidth=1)
        
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    plt.suptitle(f"Sim-to-Real Inference: {os.path.basename(TEST_SIM_CSV)}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()