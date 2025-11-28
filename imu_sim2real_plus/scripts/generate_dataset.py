import argparse, os, json, numpy as np, yaml
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
from imu_sim2real_plus.dataio.serialization import save_sequence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--out', type=str, default='runs/sim')
    ap.add_argument('--sequences', type=int, default=5)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--odr', type=int, default=400)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    os.makedirs(args.out, exist_ok=True)
    dt = 1.0/args.odr
    for i in range(args.sequences):
        N = int(args.seconds*args.odr)
        R = np.repeat(np.eye(3)[None,...], N, axis=0)
        w = np.zeros((N,3)); aW = np.zeros((N,3)); wdot = np.zeros((N,3))
        r = np.array(cfg['mount']['lever_arm_m'])
        cfg['imu']['gyro']['noise_density'][0] = float(cfg['imu']['gyro']['noise_density'][0])
        cfg['imu']['gyro']['noise_density'][1] = float(cfg['imu']['gyro']['noise_density'][1])
        cfg['imu']['accel']['noise_density'][0] = float(cfg['imu']['accel']['noise_density'][0])
        cfg['imu']['accel']['noise_density'][1] = float(cfg['imu']['accel']['noise_density'][1])
        seq, params = synth_measurements(R, w, aW, wdot, r, cfg, dt, seed=42+i)
        meta = dict(odr=args.odr, seconds=args.seconds, noise_params=params)
        save_sequence(args.out, i, seq, meta)
    print(f'Wrote {args.sequences} sequences to {args.out}')

if __name__ == '__main__':
    main()
