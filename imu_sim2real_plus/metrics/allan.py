import numpy as np
def allan_deviation(x: np.ndarray, fs: float, taus: np.ndarray):
    x = np.asarray(x).ravel()
    N = len(x)
    sigmas = np.zeros_like(taus, dtype=float)
    y = np.cumsum(x) / fs  # integral
    for i, tau in enumerate(taus):
        m = int(round(tau * fs))
        if m < 2 or 2*m >= N:
            sigmas[i] = np.nan
            continue
        diff = y[2*m:] - 2*y[m:-m] + y[:-2*m]
        sigmas[i] = np.sqrt(0.5 * np.mean(diff**2) / (tau**2))
    return sigmas
