import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib  # To save the scaler
from model_defs import IMUResidualModel

# --- CONFIGURATION ---
SEQ_LEN = 50  # Look at past 50 steps to predict current noise
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001


#Run this once to teach the model. It handles the unit conversion and "Math" required to make Sim match Real.

class SimRealDataset(Dataset):
    def __init__(self, sim_csv, real_csv):
        # 1. Load Data
        sim_df = pd.read_csv(sim_csv)
        real_df = pd.read_csv(real_csv)

        # 2. Preprocess SIM Data
        # We need to calculate Joint Velocities (Sim CSV didn't have them)
        joint_cols = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
        dt = sim_df['time'].diff().mean()
        # Create velocity columns
        sim_vels = sim_df[joint_cols].diff() / dt
        sim_vels = sim_vels.fillna(0)
        
        # 3. Preprocess REAL Data (Unit Conversion)
        # Real logs are often in 'g', Sim is in 'm/s^2'. Converting Real to m/s^2
        real_df['acc_x_sys'] = real_df['acc_x_g'] * 9.81
        real_df['acc_y_sys'] = real_df['acc_y_g'] * 9.81
        real_df['acc_z_sys'] = real_df['acc_z_g'] * 9.81
        
        # 4. Alignment
        # CRITICAL: This assumes your CSVs are already row-for-row aligned.
        # If they aren't, you must trim them here.
        min_len = min(len(sim_df), len(real_df))
        sim_df = sim_df.iloc[:min_len]
        real_df = real_df.iloc[:min_len]
        sim_vels = sim_vels.iloc[:min_len]

        # 5. Build Input Features (X)
        # 6 IMU + 7 Joints + 7 Joint Vels + 4 Quats = 24 Features
        self.feature_data = np.hstack([
            sim_df[['acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values,
            sim_df[joint_cols].values,
            sim_vels.values,
            sim_df[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values
        ])

        # 6. Build Targets (Y) -> The RESIDUAL (Error)
        # Target = Real_Reading - Sim_Clean_Reading
        real_imu = real_df[['acc_x_sys', 'acc_y_sys', 'acc_z_sys', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
        sim_imu = sim_df[['acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values
        
        self.residuals = real_imu - sim_imu

        # 7. Normalize Inputs
        self.scaler = StandardScaler()
        self.feature_data = self.scaler.fit_transform(self.feature_data)
        
        # Convert to Torch
        self.feature_data = torch.FloatTensor(self.feature_data)
        self.residuals = torch.FloatTensor(self.residuals)

    def __len__(self):
        return len(self.feature_data) - SEQ_LEN

    def __getitem__(self, idx):
        # Input: A window of history
        x = self.feature_data[idx : idx + SEQ_LEN]
        # Output: The noise at the END of that window
        y = self.residuals[idx + SEQ_LEN] 
        return x, y

def train():
    dataset = SimRealDataset("clean_sim_log.csv", "real_robot_log.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Save the scaler immediately (We need this for the live simulation!)
    joblib.dump(dataset.scaler, "imu_scaler.pkl")
    print("Scaler saved to imu_scaler.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMUResidualModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Starting Training...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "imu_noise_weights.pth")
    print("Model saved to imu_noise_weights.pth")

if __name__ == "__main__":
    train()