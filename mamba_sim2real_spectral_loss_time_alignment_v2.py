import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy import signal  # <--- Essential for alignment and smoothing
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR
import joblib

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CONFIG = {
    'data_dir': './logs_2',            
    'seq_len': 512,                  
    'input_dim': 6,                   # CHANGED: Reduced from 20 to 6 (Only Sim IMU)
    'output_dim': 6,                  
    'd_model': 256,                  
    'd_state': 64,                    
    'd_conv': 4,
    'expand': 2,
    'batch_size': 128,                
    'lr': 5e-4,                      
    'epochs': 50,                    
    'chunk_size_files': 10,          
    'scaler_path': 'scaler.pkl',
    'max_align_shift': 0.5            # Max seconds to shift for alignment
}

# ==============================================================================
# 2. DATASET (Alignment + Smooth Inputs)
# ==============================================================================

class VariableODRDataset(IterableDataset):
    def __init__(self, file_pairs, seq_len, chunk_size_files=5, do_shuffle=True, scaler=None, training=True, max_align_shift=0.5):
        super().__init__()
        self.file_pairs = file_pairs
        self.seq_len = seq_len
        self.chunk_size_files = chunk_size_files
        self.do_shuffle = do_shuffle
        self.scaler = scaler
        self.training = training
        self.max_align_shift = max_align_shift

        self.meta_chunks = [
            self.file_pairs[i : i + self.chunk_size_files]
            for i in range(0, len(self.file_pairs), self.chunk_size_files)
        ]

    def _align_timestamps(self, sim_time, sim_data, real_time, real_data):
        """
        Calculates lag using Cross-Correlation on Gyro Norm and shifts real_time.
        Prevents "phantom" variance caused by timing errors.
        """
        # 1. Resample both to a common, coarse grid for speed (100Hz)
        common_dt = 0.01
        duration = min(sim_time[-1], real_time[-1])
        t_common = np.arange(0, duration, common_dt)
       
        # We use Gyro Magnitude (last 3 columns) for robust alignment
        sim_mag = np.linalg.norm(sim_data[:, 3:], axis=1)
        real_mag = np.linalg.norm(real_data[:, 3:], axis=1)

        f_sim = interp1d(sim_time, sim_mag, bounds_error=False, fill_value=0)
        f_real = interp1d(real_time, real_mag, bounds_error=False, fill_value=0)
       
        s_resamp = f_sim(t_common)
        r_resamp = f_real(t_common)
       
        # 2. Cross-Correlation
        correlation = signal.correlate(s_resamp, r_resamp, mode='full')
        lags = signal.correlation_lags(len(s_resamp), len(r_resamp), mode='full')
       
        best_lag_idx = np.argmax(correlation)
        best_lag_steps = lags[best_lag_idx]
        time_shift = best_lag_steps * common_dt

        # 3. Sanity Check
        if abs(time_shift) > self.max_align_shift:
            return real_time

        # 4. Apply Shift: If Real is "behind" (lag positive), subtract time to move it left
        return real_time + time_shift

    def _process_pair(self, sim_csv, real_csv):
        # 1. Load Data
        sim_df = pd.read_csv(sim_csv)
        real_df = pd.read_csv(real_csv)

        # Base Time Normalization
        sim_time = sim_df['time'].values - sim_df['time'].values[0]
        real_time = real_df['time'].values - real_df['time'].values[0]

        # Extract Raw IMU for alignment
        sim_imu_raw = sim_df[['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']].values
       
        real_acc = real_df[['acc_x_g', 'acc_y_g', 'acc_z_g']].values
        real_gyro = real_df[['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
        real_imu_raw = np.hstack([real_acc, real_gyro])

        # --- FIX 1: ALIGNMENT ---
        real_time = self._align_timestamps(sim_time, sim_imu_raw, real_time, real_imu_raw)

        # ODR Augmentation
        if self.training:
            step_size = np.random.randint(1, 5)
            real_time = real_time[::step_size]
            real_df = real_df.iloc[::step_size].reset_index(drop=True)

        current_dt = np.mean(np.diff(real_time))
        if np.isnan(current_dt) or current_dt <= 0: current_dt = 0.0025

        # --- FIX 2: SMOOTH DERIVATIVES ---
        # Replaces .diff() which amplifies quantization noise
        # joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']  <-- REMOVED
        # dt_sim = np.mean(np.diff(sim_time))                        <-- REMOVED
        # if dt_sim == 0: dt_sim = 0.016                             <-- REMOVED
       
        # sim_vels_np = signal.savgol_filter(...)                    <-- REMOVED
       
        # Modified to use ONLY sim IMU readings
        sim_features_orig = sim_imu_raw

        # Interpolation (Sim to Shifted Real)
        valid_mask = (real_time >= sim_time[0]) & (real_time <= sim_time[-1])
        real_time_clipped = real_time[valid_mask]
       
        if len(real_time_clipped) < self.seq_len:
            return None, None

        # Re-extract real data to ensure we have the correct ODR slice
        # Note: real_df was already decimated above if training=True
        real_acc = real_df[['acc_x_g', 'acc_y_g', 'acc_z_g']].values
        real_gyro = real_df[['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
        real_imu_clipped = np.hstack([real_acc, real_gyro])[valid_mask]

        interpolator = interp1d(
            sim_time,
            sim_features_orig,
            axis=0,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        sim_features_resampled = interpolator(real_time_clipped)

        # Target Generation
        sim_imu = sim_features_resampled[:, :6]
        dt_col = np.full((len(sim_features_resampled), 1), current_dt)
        X = np.hstack([sim_features_resampled, dt_col])
        Y = real_imu_clipped - sim_imu                        

        return X, Y

    def __iter__(self):
        worker_info = get_worker_info()
        chunks = self.meta_chunks
        if worker_info is not None:
            chunks = [c for i, c in enumerate(chunks) if i % worker_info.num_workers == worker_info.id]
        if self.do_shuffle: np.random.shuffle(chunks)

        for meta_chunk in chunks:
            x_buf, y_buf = [], []
            for sim_path, real_path in meta_chunk:
                try:
                    feat, res = self._process_pair(sim_path, real_path)
                    if feat is None: continue
                    if self.scaler:
                        # Modified scaler indexing to match new input_dim (6)
                        feat[:, :6] = self.scaler.transform(feat[:, :6])
                    x_buf.append(feat)
                    y_buf.append(res)
                except Exception:
                    continue

            if not x_buf: continue
            x_tensor = torch.FloatTensor(np.concatenate(x_buf, axis=0))
            y_tensor = torch.FloatTensor(np.concatenate(y_buf, axis=0))
            num_samples = len(x_tensor) - self.seq_len
            if num_samples <= 0: continue

            indices = np.arange(num_samples)
            if self.do_shuffle: np.random.shuffle(indices)
            for i in indices:
                yield x_tensor[i : i+self.seq_len], y_tensor[i : i+self.seq_len]

# ==============================================================================
# 3. MAMBA MODEL (Unchanged)
# ==============================================================================

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
# 4. LOSS (Decoupled FFT)
# ==============================================================================

class Sim2RealLoss(nn.Module):
    def __init__(self, fft_weight=0.1):
        super().__init__()
        self.nll_loss = nn.GaussianNLLLoss(full=True)
        self.fft_weight = fft_weight

    def forward(self, mu, log_var, target):
        var = torch.exp(log_var)
        nll = self.nll_loss(mu, target, var)

        # --- FIX 3: FFT ON MEAN ONLY ---
        # Calculates spectral loss on the Mean (Physics) instead of the Sample (Noise)
        # This prevents the model from inflating sigma just to fake high frequencies.
        fft_pred = torch.fft.rfft(mu, dim=1)
        fft_target = torch.fft.rfft(target, dim=1)
        fft_loss = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))
       
        return nll + (self.fft_weight * fft_loss)

# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================

def get_lr_scheduler(optimizer, warmup_steps, total_training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def match_files(data_dir):
    # 1. Update search pattern to match 'imu_400hz_*'
    search_pattern = os.path.join(data_dir, "**", "imu_400hz_*.csv")
    all_files = sorted(glob.glob(search_pattern, recursive=True))
    pairs = []
    print(f"Scanning {data_dir}...")
   
    for sim_path in tqdm(all_files):
        filename = os.path.basename(sim_path)
        
        # 2. Extract ID (e.g., 'imu_400hz_251.csv' -> '251')
        file_id = filename.replace("imu_400hz_", "").replace(".csv", "")
        
        # 3. Construct the expected Real filename (e.g., 'real_400hz_251.csv')
        # We assume the real file is in the SAME folder as the sim file
        real_filename = f"real_400hz_{file_id}.csv"
        real_path = os.path.join(os.path.dirname(sim_path), real_filename)
        
        if os.path.exists(real_path):
            pairs.append((sim_path, real_path))
            
    print(f"Found {len(pairs)} pairs.")
    return pairs

def fit_scaler(file_pairs):
    print("Fitting Scaler...")
    scaler = StandardScaler()
    collected_data = []
    indices = np.random.choice(len(file_pairs), min(len(file_pairs), 20), replace=False)
   
    for idx in indices:
        sim_path, _ = file_pairs[idx]
        try:
            df = pd.read_csv(sim_path)
            # Use same smooth calculation for scaler fitting!
            # joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']  <-- REMOVED
            # dt_sim = 0.016                                             <-- REMOVED
            # vels_np = signal.savgol_filter(...)                        <-- REMOVED
           
            imu_cols = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
            # Modified to use ONLY sim IMU readings
            feats = df[imu_cols].values
           
            if len(feats) > 5000:
                feats = feats[np.random.choice(len(feats), 5000, replace=False)]
            collected_data.append(feats)
        except: pass
    scaler.fit(np.concatenate(collected_data, axis=0))
    return scaler

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Setup
    file_pairs = match_files(CONFIG['data_dir'])
    split = int(len(file_pairs) * 0.9)
    train_pairs, val_pairs = file_pairs[:split], file_pairs[split:]

    # 2. Scaler
    if os.path.exists(CONFIG['scaler_path']):
        scaler = joblib.load(CONFIG['scaler_path'])
    else:
        scaler = fit_scaler(train_pairs)
        joblib.dump(scaler, CONFIG['scaler_path'])

    # 3. Model
    model = MambaSeq2Seq(
        input_dim=CONFIG['input_dim'],
        output_dim=CONFIG['output_dim'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = Sim2RealLoss(fft_weight=0.1).to(device)

    # Scheduler
    est_steps = 2000 * CONFIG['epochs']
    warmup_steps = int(est_steps * 0.1)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, est_steps)
    print(f"Scheduler: {est_steps} steps, {warmup_steps} warmup.")

    # 4. Loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_ds = VariableODRDataset(train_pairs, CONFIG['seq_len'], scaler=scaler, training=True, max_align_shift=CONFIG['max_align_shift'])
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], num_workers=4)
       
        total_loss = 0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
       
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
           
            optimizer.zero_grad()
            mu, log_var = model(x)
            loss = criterion(mu, log_var, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
           
            scheduler.step()
           
            total_loss += loss.item()
            steps += 1
           
            curr_lr = scheduler.get_last_lr()[0]
            with torch.no_grad():
                avg_sigma = torch.exp(0.5 * log_var).mean().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{curr_lr:.6f}",
                'sigma': f"{avg_sigma:.3f}"
            })

        print(f"Ep {epoch+1}: Train {total_loss/max(1, steps):.5f}")
       
        # Validation
        val_ds = VariableODRDataset(val_pairs, CONFIG['seq_len'], scaler=scaler, training=False, do_shuffle=False, max_align_shift=CONFIG['max_align_shift'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=2)
        val_loss = 0
        val_steps = 0
        val_sigma_sum = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mu, log_var = model(x)
                loss = criterion(mu, log_var, y)
                val_loss += loss.item()
                val_steps += 1
                val_sigma_sum += torch.exp(0.5 * log_var).mean().item()
       
        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_sigma = val_sigma_sum / max(1, val_steps)

        print(f"Ep {epoch+1}: Train Loss {total_loss/max(1, steps):.5f} | Val Loss {avg_val_loss:.5f} | Val Sigma {avg_val_sigma:.3f}")
       
        torch.save(model.state_dict(), "best_model_probabilistic.pth")
