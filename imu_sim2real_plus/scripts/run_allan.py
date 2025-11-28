import argparse, numpy as np, json
from imu_sim2real_plus.metrics.allan import allan_deviation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', type=str, required=True)
    ap.add_argument('--fs', type=float, required=True)
    ap.add_argument('--taus', type=str, default='0.1,0.2,0.5,1,2,5,10,20,50,100')
    args = ap.parse_args()
    x = np.load(args.series).astype(float)
    taus = np.array([float(t) for t in args.taus.split(',')])
    s = allan_deviation(x, args.fs, taus)
    print(json.dumps({'taus': taus.tolist(), 'adev': s.tolist()}, indent=2))

if __name__ == '__main__':
    main()
