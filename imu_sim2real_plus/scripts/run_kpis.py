import argparse, glob, json, numpy as np
from imu_sim2real_plus.metrics.kpi import quat_geodesic_deg, summarize_errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_glob', type=str, required=True)
    ap.add_argument('--gt_glob', type=str, required=True)
    args = ap.parse_args()
    preds = sorted(glob.glob(args.pred_glob))
    gts = sorted(glob.glob(args.gt_glob))
    all_err = []
    for p,g in zip(preds,gts):
        qhat = np.load(p); qgt = np.load(g)
        err = quat_geodesic_deg(qgt, qhat).reshape(-1)
        all_err.append(err)
    all_err = np.concatenate(all_err) if all_err else np.array([])
    print(json.dumps(summarize_errors(all_err), indent=2))

if __name__ == '__main__':
    main()
