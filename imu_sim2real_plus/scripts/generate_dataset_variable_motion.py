
import argparse, os, json, numpy as np, yaml
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
from imu_sim2real_plus.dataio.serialization import save_sequence
from scipy.spatial.transform import Rotation as R_scipy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--out', type=str, default='runs/variable_motion_sim')
    ap.add_argument('--sequences', type=int, default=1)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--odr', type=int, default=400)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    os.makedirs(args.out, exist_ok=True)
    dt = 1.0/args.odr

    # --- Motion Generation (Variable Motion) ---
    # Parameters for variable motion (sinusoidal)
    freq_w = np.array([0.5, 0.7, 0.9]) # Frequencies for angular velocity (Hz)
    amp_w = np.deg2rad(np.array([5, 7, 9])) # Amplitudes for angular velocity (rad/s)

    freq_a = np.array([0.3, 0.6, 0.8]) # Frequencies for linear acceleration (Hz)
    amp_a = np.array([0.1, 0.15, 0.2]) # Amplitudes for linear acceleration (m/s^2)

    for i in range(args.sequences):
        N = int(args.seconds*args.odr)
        time_vec = np.arange(N) * dt

        # Ground Truth Angular Velocity (w_B) - Sinusoidal
        w_B = np.zeros((N, 3))
        for j in range(3):
            w_B[:, j] = amp_w[j] * np.sin(2 * np.pi * freq_w[j] * time_vec)

        # Ground Truth Rotation Matrix (R_WB)
        R_WB = np.zeros((N, 3, 3))
        current_rotation = R_scipy.from_euler('xyz', [0, 0, 0], degrees=False)
        for k in range(N):
            delta_angle = w_B[k] * dt
            delta_rotation = R_scipy.from_rotvec(delta_angle)
            current_rotation = current_rotation * delta_rotation
            R_WB[k] = current_rotation.as_matrix()

        # Ground Truth Linear Acceleration (a_W) - Sinusoidal
        a_W = np.zeros((N, 3))
        for j in range(3):
            a_W[:, j] = amp_a[j] * np.sin(2 * np.pi * freq_a[j] * time_vec + np.pi/4) # Add phase shift

        # Ground Truth Angular Acceleration Derivative (wdot_B)
        # For sinusoidal w_B, wdot_B is also sinusoidal (derivative of sine is cosine)
        wdot_B = np.zeros((N, 3))
        for j in range(3):
            wdot_B[:, j] = amp_w[j] * (2 * np.pi * freq_w[j]) * np.cos(2 * np.pi * freq_w[j] * time_vec)

        r = np.array(cfg['mount']['lever_arm_m'])
        cfg['imu']['gyro']['noise_density'][0] = float(cfg['imu']['gyro']['noise_density'][0])
        cfg['imu']['gyro']['noise_density'][1] = float(cfg['imu']['gyro']['noise_density'][1])
        cfg['imu']['accel']['noise_density'][0] = float(cfg['imu']['accel']['noise_density'][0])
        cfg['imu']['accel']['noise_density'][1] = float(cfg['imu']['accel']['noise_density'][1])

        # Synthesize measurements with motion
        seq = synth_measurements(R_WB, w_B, a_W, wdot_B, r, cfg, dt, seed=42+i)
        meta = dict(odr=args.odr, seconds=args.seconds)

        # Save ground truth motion data along with measurements
        seq['gt_R_WB'] = R_WB
        seq['gt_w_B'] = w_B
        seq['gt_a_W'] = a_W
        seq['gt_wdot_B'] = wdot_B # Save wdot_B as well

        save_sequence(args.out, i, seq, meta)
    print(f'Wrote {args.sequences} sequences to {args.out}')

if __name__ == '__main__':
    main()
