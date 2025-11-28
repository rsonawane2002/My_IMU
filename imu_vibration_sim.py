from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
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

def simulate_imu_with_vibration(
        t, true_w_B, true_a_B, rpm_profile,
        modes_accel, modes_gyro,
        accel_white_std=0.03, gyro_white_std=0.002,
        gyro_bias_rw_std=2e-5, g_sensitivity=0.002,
        quantize_accel_mg=None, quantize_gyro_dps=None, seed=123):
    fs = 1.0/np.mean(np.diff(t))
    exc = synth_motor_excitation(t, rpm_profile, harmonics={1:1.0,2:0.35,3:0.2})
    floor = arma_colored_noise(t, sigma=0.4, ar=0.96, ma=0.2, seed=seed)
    base_exc = exc + 0.25*floor

    axis_scale = np.array([1.0,0.6,0.9])

    vib_accel = np.zeros_like(true_a_B)
    vib_gyro  = np.zeros_like(true_w_B)
    for ax in range(3):
        vib_accel[:,ax] = axis_scale[ax] * apply_filter_bank(base_exc, fs, [m for m in modes_accel if ax in m.axes])
        vib_gyro[:,ax]  = axis_scale[ax] * apply_filter_bank(base_exc, fs, [m for m in modes_gyro  if ax in m.axes])

    gyro_coupling = g_sensitivity * vib_accel

    rng = np.random.default_rng(seed+1)
    n_acc = rng.normal(0, accel_white_std, size=true_a_B.shape)
    n_gyr = rng.normal(0, gyro_white_std,  size=true_w_B.shape)

    dt = np.mean(np.diff(t))
    b = np.zeros_like(true_w_B)
    for i in range(1, len(t)):
        b[i] = b[i-1] + rng.normal(0, gyro_bias_rw_std*np.sqrt(dt), size=3)

    a_meas = true_a_B + vib_accel + n_acc
    w_meas = true_w_B + vib_gyro + gyro_coupling + n_gyr + b

    if quantize_accel_mg is not None and quantize_accel_mg > 0:
        q = quantize_accel_mg * 9.80665 / 1000.0
        a_meas = np.round(a_meas / q) * q
    if quantize_gyro_dps is not None and quantize_gyro_dps > 0:
        q = np.deg2rad(quantize_gyro_dps)
        w_meas = np.round(w_meas / q) * q

    return w_meas, a_meas, {'excitation': base_exc, 'gyro_bias': b}
