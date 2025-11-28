from typing import Callable, Dict
import numpy as np

def gather_truth(get_state_fn: Callable[[], Dict[str, np.ndarray]], seconds: float, odr_hz: int):
    N = int(seconds * odr_hz)
    R = np.zeros((N,3,3)); W = np.zeros((N,3)); A = np.zeros((N,3)); WD = np.zeros((N,3))
    dt = 1.0/odr_hz
    for k in range(N):
        s = get_state_fn()
        R[k] = s['R_WB']; W[k] = s['w_B']; A[k] = s['a_W']; WD[k] = s['wdot_B']
    return R, W, A, WD, dt
