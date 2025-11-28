from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import signal

@dataclass
class ResonantMode:
    f0: float
    zeta: float
    gain: float
    axes: Tuple[int, ...] = (0,1,2)

def bilinear_resonator(fs: float, mode: ResonantMode):
    wn = 2*np.pi*mode.f0
    zeta = mode.zeta
    num = [0, 2*zeta*wn, 0]
    den = [1, 2*zeta*wn, wn**2]
    num = [c*mode.gain for c in num]
    bz, az = signal.bilinear(num, den, fs=fs)
    return np.array(bz, dtype=float), np.array(az, dtype=float)

def synth_motor_excitation(t, rpm, harmonics=None, phase0=0.0):
    if harmonics is None:
        harmonics = {1:1.0, 2:0.4, 3:0.2}
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0
    phi = np.cumsum(2*np.pi*(rpm/60.0)*dt) + phase0
    x = np.zeros_like(t)
    rng = np.random.default_rng(0)
    for h, amp in harmonics.items():
        x += amp * np.sin(h * phi + 2*np.pi*rng.random())
    return x

def arma_colored_noise(t, sigma=1.0, ar=0.95, ma=0.2, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(len(t)) * sigma
    y = np.zeros_like(t)
    for i in range(1, len(t)):
        y[i] = ar * y[i-1] + e[i] + ma * e[i-1]
    return y

def apply_filter_bank(x, fs, modes: List[ResonantMode]):
    y = np.zeros_like(x)
    for m in modes:
        b, a = bilinear_resonator(fs, m)
        y += signal.lfilter(b, a, x)
    return y
