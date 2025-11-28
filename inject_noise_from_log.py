import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import argparse
import numpy as np
import yaml
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
from imu_sim2real_plus.dataio.serialization import save_sequence

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=True, help='Path to the trajectory.csv log file')
    parser.add_argument('--config', type=str, required=True, help='Path to the IMU configuration file')
    parser.add_argument('--out', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--lever_arm', nargs=3, type=float, metavar=('LX','LY','LZ'), default=[0.0, 0.0, 0.01],
                        help='IMU lever arm in meters in the body frame (default: 0 0 0.01)')
    parser.add_argument('--disable_noise_logging', action='store_true',
                        help='If set, do not include noise parameters in the saved meta.json')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for noise synthesis (default: 12345)')
    parser.add_argument('--scale_a_min', type=float, default=0.99, help='Accel scale min (default: 0.99)')
    parser.add_argument('--scale_a_max', type=float, default=1.01, help='Accel scale max (default: 1.01)')
    parser.add_argument('--dump_model_csv', action='store_true', help='Also dump time, f_meas, w_meas to a CSV in the output dir')
    parser.add_argument('--csv_out', type=str, default=None, help='Optional path to write model output CSV (time,f_meas,w_meas)')
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Explicitly cast noise_density values to float
    cfg['imu']['gyro']['noise_density'][0] = float(cfg['imu']['gyro']['noise_density'][0])
    cfg['imu']['gyro']['noise_density'][1] = float(cfg['imu']['gyro']['noise_density'][1])
    cfg['imu']['accel']['noise_density'][0] = float(cfg['imu']['accel']['noise_density'][0])
    cfg['imu']['accel']['noise_density'][1] = float(cfg['imu']['accel']['noise_density'][1])
    # Enforce requested accelerometer scale range
    cfg['imu']['accel']['scale_range'] = [float(args.scale_a_min), float(args.scale_a_max)]

    # Read the log file
    with open(args.log_file, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    data = np.array(data, dtype=float)

    # Get column indices
    acc_x_idx = header.index('acc_x')
    acc_y_idx = header.index('acc_y')
    acc_z_idx = header.index('acc_z')
    ang_vel_x_idx = header.index('ang_vel_x')
    ang_vel_y_idx = header.index('ang_vel_y')
    ang_vel_z_idx = header.index('ang_vel_z')
    quat_w_idx = header.index('quat_w')
    quat_x_idx = header.index('quat_x')
    quat_y_idx = header.index('quat_y')
    quat_z_idx = header.index('quat_z')

    # Extract data
    # Re-interpreting acc_x, acc_y, acc_z as linear acceleration (a_B) instead of specific force (f_B)
    a_B_input = data[:, [acc_x_idx, acc_y_idx, acc_z_idx]]
    w_B = data[:, [ang_vel_x_idx, ang_vel_y_idx, ang_vel_z_idx]]
    quats = data[:, [quat_w_idx, quat_x_idx, quat_y_idx, quat_z_idx]]
    # Time column may not be first; locate by header name
    try:
        time_idx = header.index('time')
    except ValueError:
        # Fallback to first column if no explicit 'time' header
        time_idx = 0
    t = data[:, time_idx]
    dt = np.mean(np.diff(t))

    # Convert quaternions to rotation matrices
    # Interpret quaternion as world->body (NED aerospace convention).
    # Build R_WB (world -> body). Its transpose R_BW maps body -> world.
    R_WB = np.array([quaternion_to_rotation_matrix(q) for q in quats])
    R_BW = R_WB.transpose(0, 2, 1)

    # Define gravity vector in world frame
    g_W = np.array([0, 0, 9.81])

    # Specific force in body frame from linear acceleration (body):
    # g_B = R_WB * g_W,  f_B = a_B - g_B
    g_B = R_WB @ g_W
    f_B_true_input = a_B_input - g_B

    # Calculate a_W to pass to synth_measurements such that it correctly calculates specific force
    # World linear acceleration that reproduces the same specific force:
    # a_W = R_BW * f_B + g_W
    a_W_to_pass = (R_BW @ f_B_true_input[..., None]).squeeze() + g_W

    # Calculate wdot_B
    wdot_B = np.gradient(w_B, dt, axis=0)

    # Lever arm from CLI (body frame)
    r_lever = np.array(args.lever_arm, dtype=float)

    # Synthesize IMU data
    # The synthesizer expects R parameter as body->world; pass R_BW.
    seq, params = synth_measurements(R_BW, w_B, a_W_to_pass, wdot_B, r_lever, cfg, dt, seed=args.seed)

    # Add ground truth data to the sequence dictionary
    seq['gt_a_W'] = a_W_to_pass # This is the a_W that was passed to synth_measurements
    seq['gt_w_B'] = w_B
    # Save world->body rotation for downstream plotting
    seq['gt_R_WB'] = R_WB
    seq['input_a_B'] = a_B_input # Save the original input linear acceleration
    seq['input_specific_force_B'] = f_B_true_input # Save the calculated input specific force

    # Save the sequence
    meta = {
        'odr': int(1/dt),
        'seconds': t[-1] - t[0],
        'config': args.config,
        'log_file': args.log_file
    }
    if not args.disable_noise_logging:
        meta['noise_params'] = params
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    save_sequence(args.out, 0, seq, meta)
    # Optionally dump params as CSV for convenience
    try:
        import json, csv
        params_csv = os.path.join(args.out, 'noise_params.csv')
        with open(params_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['key','value'])
            def emit(prefix, obj):
                if isinstance(obj, dict):
                    for k,v in obj.items():
                        emit(f"{prefix}.{k}" if prefix else k, v)
                else:
                    w.writerow([prefix, json.dumps(obj)])
            emit('', params)
    except Exception as e:
        print(f"Warning: failed to write noise_params.csv: {e}")

    # Optionally dump model output to CSV
    if args.dump_model_csv or args.csv_out:
        import csv
        out_csv = args.csv_out or os.path.join(args.out, 'seq_00000_model_output.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time','f_x','f_y','f_z','w_x','w_y','w_z'])
            for ti, fv, wv in zip(t, seq['f_meas'], seq['w_meas']):
                writer.writerow([f"{ti:.9f}", f"{fv[0]:.9f}", f"{fv[1]:.9f}", f"{fv[2]:.9f}", f"{wv[0]:.9f}", f"{wv[1]:.9f}", f"{wv[2]:.9f}"])
    print(f"Wrote sequence to {args.out}")

if __name__ == '__main__':
    main()
