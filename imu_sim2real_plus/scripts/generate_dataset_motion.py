
import argparse, os, json, numpy as np, yaml
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
from imu_sim2real_plus.dataio.serialization import save_sequence
from scipy.spatial.transform import Rotation as R_scipy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--out', type=str, default='runs/sim')
    ap.add_argument('--sequences', type=int, default=1)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--odr', type=int, default=400)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    os.makedirs(args.out, exist_ok=True)
    dt = 1.0/args.odr

    # --- Motion Generation (Circular Motion) ---
    # Parameters for circular motion
    radius = 1.0 # meters
    angular_speed = np.deg2rad(30) # 30 degrees per second around Z-axis

    for i in range(args.sequences):
        N = int(args.seconds*args.odr)
        time_vec = np.arange(N) * dt

        # Ground Truth Angular Velocity (w_B)
        w_B = np.zeros((N, 3))
        w_B[:, 2] = angular_speed # Constant rotation around Z-axis

        # Ground Truth Rotation Matrix (R_WB)
        # Integrate angular velocity to get rotation
        R_WB = np.zeros((N, 3, 3))
        current_rotation = R_scipy.from_euler('xyz', [0, 0, 0], degrees=False)
        for k in range(N):
            delta_angle = w_B[k] * dt
            delta_rotation = R_scipy.from_rotvec(delta_angle)
            current_rotation = current_rotation * delta_rotation
            R_WB[k] = current_rotation.as_matrix()

        # Ground Truth Linear Acceleration (a_W) - Centripetal acceleration
        a_W = np.zeros((N, 3))
        # Assuming motion in XY plane, centripetal acceleration points towards center
        # For constant angular speed, magnitude is omega^2 * r
        # Direction is -x for motion along +y, -y for motion along +x
        # This is simplified for a fixed radius and constant angular speed
        # A more accurate model would involve position and velocity
        # For simplicity, let's assume the IMU is at [radius, 0, 0] initially and rotates around Z
        # Then a_W = -omega^2 * position_vector
        # Here, we'll just make it a constant centripetal acceleration for demonstration
        # This is a simplification, a_W should be derived from actual trajectory
        # For a simple circular motion at constant speed, a_W is constant magnitude, rotating direction
        # Let's make it a simple sinusoidal acceleration for demonstration
        a_W[:, 0] = -radius * (angular_speed**2) * np.cos(angular_speed * time_vec)
        a_W[:, 1] = -radius * (angular_speed**2) * np.sin(angular_speed * time_vec)

        # Ground Truth Angular Acceleration Derivative (wdot_B)
        wdot_B = np.zeros((N, 3)) # Constant angular velocity, so derivative is zero

        # --- RPM Profile Generation ---
        rpm_profile = 800 + 600 * np.sin(2 * np.pi * 0.1 * time_vec)

        r = np.array(cfg['mount']['lever_arm_m'])
        cfg['imu']['gyro']['noise_density'][0] = float(cfg['imu']['gyro']['noise_density'][0])
        cfg['imu']['gyro']['noise_density'][1] = float(cfg['imu']['gyro']['noise_density'][1])
        cfg['imu']['accel']['noise_density'][0] = float(cfg['imu']['accel']['noise_density'][0])
        cfg['imu']['accel']['noise_density'][1] = float(cfg['imu']['accel']['noise_density'][1])

        # Synthesize measurements with motion
        seq = synth_measurements(R_WB, w_B, a_W, wdot_B, r, cfg, dt, seed=42+i, rpm_profile=rpm_profile)
        meta = dict(odr=args.odr, seconds=args.seconds)

        # Save ground truth motion data along with measurements
        seq['gt_R_WB'] = R_WB
        seq['gt_w_B'] = w_B
        seq['gt_a_W'] = a_W

        save_sequence(args.out, i, seq, meta)
    print(f'Wrote {args.sequences} sequences to {args.out}')

if __name__ == '__main__':
    main()
