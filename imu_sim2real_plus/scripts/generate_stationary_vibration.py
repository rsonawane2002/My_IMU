import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, json, numpy as np, yaml
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
from imu_sim2real_plus.dataio.serialization import save_sequence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='imu_sim2real_plus/config/example_config.yaml')
    ap.add_argument('--out', type=str, default='runs/stationary_vibration')
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--odr', type=int, default=400)
    ap.add_argument('--rpm', type=float, default=1000.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    os.makedirs(args.out, exist_ok=True)
    dt = 1.0/args.odr
    
    N = int(args.seconds*args.odr)
    R = np.repeat(np.eye(3)[None,...], N, axis=0)
    w = np.zeros((N,3)); aW = np.zeros((N,3)); wdot = np.zeros((N,3))
    r = np.array(cfg['mount']['lever_arm_m'])
    cfg['imu']['gyro']['noise_density'][0] = float(cfg['imu']['gyro']['noise_density'][0])
    cfg['imu']['gyro']['noise_density'][1] = float(cfg['imu']['gyro']['noise_density'][1])
    cfg['imu']['accel']['noise_density'][0] = float(cfg['imu']['accel']['noise_density'][0])
    cfg['imu']['accel']['noise_density'][1] = float(cfg['imu']['accel']['noise_density'][1])
    
    rpm_profile = np.full(N, args.rpm)
    
    seq, params = synth_measurements(R, w, aW, wdot, r, cfg, dt, seed=42, rpm_profile=rpm_profile)
    meta = dict(odr=args.odr, seconds=args.seconds, rpm=args.rpm, noise_params=params)
    save_sequence(args.out, 0, seq, meta)
    print(f'Wrote sequence to {args.out}')

if __name__ == '__main__':
    main()
