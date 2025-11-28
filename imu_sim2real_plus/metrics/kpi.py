import numpy as np
def quat_geodesic_deg(q_gt: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    def norm(q): return q / np.linalg.norm(q, axis=-1, keepdims=True)
    q1 = norm(q_gt); q2 = norm(q_hat)
    dots = np.sum(q1*q2, axis=-1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return (2*np.arccos(dots)) * 180/np.pi

def summarize_errors(err_deg: np.ndarray):
    return {
        'median_deg': float(np.nanmedian(err_deg)),
        'p95_deg': float(np.nanpercentile(err_deg, 95)),
        'max_deg': float(np.nanmax(err_deg))
    }
