import torch
import torch.nn as nn


#Definition of model architecture that will be used for training. import in live/offline training script

class IMUResidualModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=6, num_layers=2):
        """
        LSTM Model for Temporal Noise Prediction.
        input_dim=24: (6 IMU + 7 Joint Pos + 7 Joint Vel + 4 Quat)
        output_dim=6: (3 Accel Noise + 3 Gyro Noise)
        """
        super(IMUResidualModel, self).__init__()
        
        # LSTM layer: Captures history (vibration buildup, drift)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Fully Connected layer: Maps abstract features to specific sensor noise
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        lstm_out, _ = self.lstm(x)
        
        # We only care about the hidden state at the LAST timestep of the window
        last_step = lstm_out[:, -1, :] 
        
        out = self.fc(last_step)
        return out