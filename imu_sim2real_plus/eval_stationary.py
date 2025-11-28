
import torch
from imu_sim2real_plus.models.baseline_net import BaselineIMUOrientation, rot6d_to_quat
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # Load model
    model = BaselineIMUOrientation()
    model.load_state_dict(torch.load('runs/stationary_model.pth'))
    model.eval()

    # Load data
    data_dir = 'runs/stationary_vibration'
    f_meas = np.load(os.path.join(data_dir, 'seq_00000_f_meas.npy'))
    w_meas = np.load(os.path.join(data_dir, 'seq_00000_w_meas.npy'))
    x = torch.from_numpy(np.concatenate([f_meas, w_meas], axis=1)).float().unsqueeze(0) # Add batch dimension

    # Run model
    with torch.no_grad():
        r6 = model(x)
        q_pred = rot6d_to_quat(r6).squeeze(0).numpy()

    # Plot
    time = np.arange(q_pred.shape[0]) / 400.0 # Assuming ODR is 400
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, f_meas, label=['fx', 'fy', 'fz'])
    plt.plot(time, w_meas, label=['wx', 'wy', 'wz'])
    plt.title('Input IMU Data with Vibration')
    plt.xlabel('Time (s)')
    plt.ylabel('Sensor Readings')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, q_pred, label=['qw', 'qx', 'qy', 'qz'])
    plt.title('Output Quaternion from Sim2Real Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('runs/stationary_vibration_output.png')
    plt.show()

if __name__ == '__main__':
    main()
