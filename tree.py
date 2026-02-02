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
# Note: GradScaler and autocast removed for stability (Fix #2)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR
import joblib

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. ROBUST CONFIGURATION
# ==============================================================================
CONFIG = {
    'data_dir': './logs_dummy',            
    
    # Data Params
    'seq_len': 512,                  # INCREASED: Better for capturing drift history
    'input_dim': 24,                 
    'output_dim': 6,                 
    
    # Model Params
    'd_model': 256,                  # INCREASED: More capacity
    'd_state': 64,                   # INCREASED: Better complex noise memory
    'd_conv': 4,
    'expand': 2,
    
    # Training Params
    'batch_size': 128,               # REDUCED: For stability with larger seq_len
    'lr': 1e-4,                      # LOWERED: Mamba prefers lower LR
    'epochs': 50,                    
    'chunk_size_files': 10,          
    'scaler_path': 'scaler.pkl',
    'alpha_start': 0.0,              # Start simple (value loss only)
    'alpha_end': 0.5                 # Gradually add derivative loss
}

# ==============================================================================
# 2. DATASET (With Column & Interpolation Fixes)
# ==============================================================================

class VariableODRDataset(IterableDataset):
    def __init__(self, file_pairs, seq_len, chunk_size_files=5, do_shuffle=True, scaler=None, training=True):
        super().__init__()
        self.file_pairs = file_pairs
        self.seq_len = seq_len
        self.chunk_size_files = chunk_size_files
        self.do_shuffle = do_shuffle
        self.scaler = scaler
        self.training = training 

        self.meta_chunks = [
            self.file_pairs[i : i + self.chunk_size_files] 
            for i in range(0, len(self.file_pairs), self.chunk_size_files)
        ]

    def _process_pair(self, sim_csv, real_csv):
        # 1. Load Data
        sim_df = pd.read_csv(sim_csv)
        real_df = pd.read_csv(real_csv)

        # Zero-reference Time
        sim_time = sim_df['time'].values - sim_df['time'].values[0]
        real_time = real_df['time_sec'].values - real_df['time_sec'].values[0]

        # ------------------------------------------------------------------
        # A. ODR AUGMENTATION
        # ------------------------------------------------------------------
        if self.training:
            step_size = np.random.randint(1, 10)
            real_time = real_time[::step_size]
            real_df = real_df.iloc[::step_size].reset_index(drop=True)

        current_dt = np.mean(np.diff(real_time))
        if np.isnan(current_dt) or current_dt <= 0: current_dt = 0.0025

        # ------------------------------------------------------------------
        # B. SIMULATION PHYSICS
        # ------------------------------------------------------------------
        joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
        dt_sim = np.mean(np.diff(sim_time))
        if dt_sim == 0: dt_sim = 0.016 
        
        sim_vels = sim_df[joint_cols].diff().fillna(0) / dt_sim

        # Correct Column Names
        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']

        sim_features_orig = np.hstack([
            sim_df[imu_cols].values,
            sim_df[joint_cols].values,
            sim_vels.values,
            sim_df[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values
        ])

        # ------------------------------------------------------------------
        # C. INTERPOLATION (Robust)
        # ------------------------------------------------------------------
        valid_mask = real_time <= sim_time[-1]
        real_time_clipped = real_time[valid_mask]
        real_df = real_df.iloc[valid_mask].reset_index(drop=True)

        if len(real_time_clipped) < self.seq_len:
            return None, None

        # FIX: extrapolate to prevent NaNs at edges
        interpolator = interp1d(
            sim_time, 
            sim_features_orig, 
            axis=0, 
            kind='linear', 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        sim_features_resampled = interpolator(real_time_clipped)

        # Normalize Quaternions
        quats = sim_features_resampled[:, -4:]
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sim_features_resampled[:, -4:] = quats / norms

        # ------------------------------------------------------------------
        # D. TARGET GENERATION
        # ------------------------------------------------------------------
        real_acc = real_df[['acc_x_g', 'acc_y_g', 'acc_z_g']].values
        real_gyro = real_df[['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
        real_imu = np.hstack([real_acc, real_gyro])
        
        sim_imu = sim_features_resampled[:, :6]

        dt_col = np.full((len(sim_features_resampled), 1), current_dt)
        X = np.hstack([sim_features_resampled, dt_col]) 
        Y = real_imu - sim_imu                          

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
                        feat[:, :24] = self.scaler.transform(feat[:, :24])
                    
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
# 3. PHYSICS-INFORMED MAMBA MODEL (Corrected Dimensions)
# ==============================================================================

class PhysicsMambaBlock(nn.Module):
    def __init__(
        self, 
        d_model, 
        d_state=16, 
        d_conv=4, 
        expand=2, 
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) 
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.bias._no_reinit = True
        
        A = repeat(
            np.arange(1, self.d_state + 1, dtype=np.float32), 
            "n -> d n", d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(torch.tensor(A)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, dt_vals):
        batch, seq, _ = x.shape

        # A. Project Inputs -> (B, L, 2*D)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (B, L, D)

        # B. Conv1D -> Transpose to (B, D, L) for Conv
        x = x.transpose(1, 2) 
        x = self.conv1d(x)[:, :, :seq]
        x = self.activation(x)
        
        # C. Compute SSM Parameters
        # x_proj needs (B, L, D) view
        x_for_proj = x.transpose(1, 2) 
        x_dbl = self.x_proj(x_for_proj) 
        
        _, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Transpose B, C to (B, State, L) for Scan
        B = B.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()
        
        # D. FORCE DT
        dt = self.dt_proj(dt_vals) # (B, L, D)
        dt = dt.transpose(1, 2).contiguous() # (B, D, L)

        # E. Gate Z -> (B, D, L)
        z = z.transpose(1, 2).contiguous()

        # F. Selective Scan (All inputs are B, Dim, L)
        y = selective_scan_fn(
            x,              
            dt,             
            A=-torch.exp(self.A_log.float()), 
            B=B,            
            C=C,            
            D=self.D.float(),
            z=z,            
            delta_bias=None, 
            delta_softplus=True,
            return_last_state=False,
        )

        # G. Output Project -> Transpose back to (B, L, D)
        y = y.transpose(1, 2)
        return self.out_proj(y)


class MambaSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_state, d_conv, expand):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.mamba_block = PhysicsMambaBlock(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x_full):
        x_features = x_full[:, :, :-1] 
        dt_vals = x_full[:, :, -1:]    
        x_emb = self.input_projection(x_features)
        mamba_out = self.mamba_block(x_emb, dt_vals)
        output = self.output_projection(mamba_out)
        return output

# ==============================================================================
# 4. LOSS & HELPERS
# ==============================================================================

class IMUSeqLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=beta, reduction='mean')

    def forward(self, y_pred, y_true, alpha=0.5):
        val_loss = self.huber(y_pred, y_true)
        pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]
        diff_loss = F.l1_loss(pred_diff, true_diff)
        return (1 - alpha) * val_loss + alpha * diff_loss

def match_files(data_dir):
    search_pattern = os.path.join(data_dir, "**", "clean_*_with_gravity.csv")
    sim_files = sorted(glob.glob(search_pattern, recursive=True))
    
    pairs = []
    print(f"Scanning {data_dir} recursively...")
    
    for sim_path in tqdm(sim_files):
        directory = os.path.dirname(sim_path)
        filename = os.path.basename(sim_path)
        file_id = filename.replace("clean_", "").replace("_with_gravity.csv", "")
        real_filename = f"real_imu_{file_id}.csv"
        real_path = os.path.join(directory, real_filename)
        
        if os.path.exists(real_path):
            pairs.append((sim_path, real_path))

    print(f"Found {len(pairs)} matched pairs.")
    return pairs

def fit_scaler(file_pairs):
    print("Fitting Scaler on subset...")
    scaler = StandardScaler()
    collected_data = []
    indices = np.random.choice(len(file_pairs), min(len(file_pairs), 20), replace=False)
    
    for idx in indices:
        sim_path, _ = file_pairs[idx]
        try:
            df = pd.read_csv(sim_path)
            
            joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
            # Correct columns
            required_cols = joint_cols + ['acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'quat_x']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"❌ Missing columns in {sim_path}: {missing}")
                continue

            vels = df[joint_cols].diff().fillna(0).values
            imu_cols = ['acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
            
            feats = np.hstack([
                df[imu_cols].values,
                df[joint_cols].values,
                vels,
                df[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values
            ])
            
            if len(feats) > 5000:
                feats = feats[np.random.choice(len(feats), 5000, replace=False)]
            collected_data.append(feats)
        except Exception as e:
            print(f"❌ Error reading {sim_path}: {e}")
            pass

    if not collected_data: raise ValueError("No data found to fit scaler.")
    scaler.fit(np.concatenate(collected_data, axis=0))
    print("Scaler fitted.")
    return scaler

def get_lr_scheduler(optimizer, warmup_steps, total_training_steps):
    """
    Linear Warmup -> Cosine Decay
    """
    def lr_lambda(current_step):
        # 1. Linear Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. Cosine Decay
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

# ==============================================================================
# 5. TRAINING LOOP (Robust Float32 + Gradient Clipping)
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Sequence-to-Sequence training on {device}")

    # 1. Setup Data
    file_pairs = match_files(CONFIG['data_dir'])
    if not file_pairs: raise FileNotFoundError("No files found.")

    split = int(len(file_pairs) * 0.9)
    train_pairs = file_pairs[:split]
    val_pairs = file_pairs[split:]

    # 2. Scaler
    if os.path.exists(CONFIG['scaler_path']):
        scaler = joblib.load(CONFIG['scaler_path'])
    else:
        scaler = fit_scaler(train_pairs)
        joblib.dump(scaler, CONFIG['scaler_path'])

    # 3. Dataloaders
    train_ds = VariableODRDataset(train_pairs, CONFIG['seq_len'], scaler=scaler, training=True)
    val_ds = VariableODRDataset(val_pairs, CONFIG['seq_len'], scaler=scaler, training=False, do_shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], num_workers=6)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=2)

    # 4. Model
    model = MambaSeq2Seq(
        input_dim=CONFIG['input_dim'],
        output_dim=CONFIG['output_dim'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand']
    ).to(device)

    criterion = IMUSeqLoss(beta=1.0)
    
    # Increase base LR slightly since we now have a scheduler to control it
    optimizer = optim.AdamW(model.parameters(), lr=5e-4) 
    
    # --- NEW: SCHEDULER SETUP ---
    # Based on your logs, you have approx 1300-1500 steps per epoch.
    steps_per_epoch_est = 1500 
    total_steps = steps_per_epoch_est * CONFIG['epochs']
    warmup_steps = int(total_steps * 0.1) # 10% warmup

    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    print(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps.")
    # ----------------------------

    # 5. Loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        steps = 0
        
        alpha = CONFIG['alpha_start'] + (CONFIG['alpha_end'] - CONFIG['alpha_start']) * (epoch / CONFIG['epochs'])
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (alpha={alpha:.2f})")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if torch.isnan(x).any() or torch.isinf(x).any():
                continue

            optimizer.zero_grad()
            
            # Standard Float32 Forward
            pred = model(x)
            loss = criterion(pred, y, alpha=alpha)

            if torch.isnan(loss):
                print("!! Loss is NaN. Skipping !!")
                optimizer.zero_grad()
                continue

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # --- NEW: STEP SCHEDULER ---
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # ---------------------------

            total_loss += loss.item()
            steps += 1
            
            # Show LR in progress bar
            pbar.set_postfix({'loss': f"{loss.item():.5f}", 'lr': f"{current_lr:.6f}"})

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                if not torch.isnan(pred).any():
                    val_loss += criterion(pred, y, alpha=0.5).item()
                    val_steps += 1
        
        avg_train = total_loss / max(1, steps) if steps > 0 else 0.0
        avg_val = val_loss / max(1, val_steps) if val_steps > 0 else 0.0
        
        print(f"Summary Ep {epoch+1}: Train {avg_train:.5f} | Val {avg_val:.5f}")
        torch.save(model.state_dict(), "best_model_seq2seq.pth")