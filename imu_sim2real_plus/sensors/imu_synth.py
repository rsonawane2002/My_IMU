import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from .vibration_model import ResonantMode, synth_motor_excitation, arma_colored_noise, apply_filter_bank

@dataclass
class GM1:
    tau: float
    sigma_rw: float
    x: np.ndarray

    def step(self, dt: float):
        self.x = self.x + (-(dt/self.tau))*self.x + self.sigma_rw*np.sqrt(dt)*np.random.randn(3)
        return self.x

def quantize(x, q):
    return np.round(x / q) * q

def synth_measurements(R_WB, w_B, a_W, wdot_B, r_lever, cfg: Dict, dt: float, seed: int = 0, rpm_profile: np.ndarray = None, base_excitation_signal: np.ndarray = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    np.random.seed(seed)
    N = R_WB.shape[0]
    I = np.eye(3)
    # Misalignment and scale
    mis_pct = np.random.uniform(cfg['imu']['misalignment_pct'][0], cfg['imu']['misalignment_pct'][1]) / 100.0
    M_a = mis_pct * (np.random.randn(3,3)*0.2)
    M_g = mis_pct * (np.random.randn(3,3)*0.2)
    # Accelerometer scale: prefer explicit range if provided, else fall back to ppm
    accel_cfg = cfg['imu']['accel']
    if 'scale_range' in accel_cfg:
        S_a = float(np.random.uniform(accel_cfg['scale_range'][0], accel_cfg['scale_range'][1]))
    else:
        S_a = 1.0 + 1e-6*np.random.uniform(accel_cfg['scale_ppm'][0], accel_cfg['scale_ppm'][1])
    S_g = 1.0 + 1e-6*np.random.uniform(cfg['imu']['gyro']['scale_ppm'][0], cfg['imu']['gyro']['scale_ppm'][1])
    # Biases
    b_a0 = np.random.uniform(cfg['imu']['accel']['bias_init'][0], cfg['imu']['accel']['bias_init'][1], size=3)
    b_g0 = np.random.uniform(cfg['imu']['gyro']['bias_init'][0], cfg['imu']['gyro']['bias_init'][1], size=3)
    tau_a = np.random.uniform(cfg['imu']['accel']['bias_tau_s'][0], cfg['imu']['accel']['bias_tau_s'][1])
    tau_g = np.random.uniform(cfg['imu']['gyro']['bias_tau_s'][0], cfg['imu']['gyro']['bias_tau_s'][1])
    na = np.random.uniform(cfg['imu']['accel']['noise_density'][0], cfg['imu']['accel']['noise_density'][1])
    ng = np.random.uniform(cfg['imu']['gyro']['noise_density'][0], cfg['imu']['gyro']['noise_density'][1])
    gm_a = GM1(tau=tau_a, sigma_rw=na, x=b_a0.copy())
    gm_g = GM1(tau=tau_g, sigma_rw=ng, x=b_g0.copy())

    # Vibration model
    vib_cfg = cfg.get('vibration')
    vib_accel = np.zeros((N,3)); vib_gyro = np.zeros((N,3)); gyro_coupling = np.zeros((N,3))
    if vib_cfg:
        base_exc = None
        fs = 1.0/dt
        if base_excitation_signal is not None:
            if base_excitation_signal.shape[0] != N:
                raise ValueError(f"Length of base_excitation_signal ({base_excitation_signal.shape[0]}) must be equal to the number of samples N ({N}).")
            base_exc = base_excitation_signal
        elif rpm_profile is not None:
            t = np.arange(N) * dt
            exc = synth_motor_excitation(t, rpm_profile, harmonics=vib_cfg['motor_harmonics'])
            floor = arma_colored_noise(t, sigma=vib_cfg['floor_noise_sigma'], ar=vib_cfg['floor_noise_ar'], ma=vib_cfg['floor_noise_ma'], seed=seed)
            base_exc = exc + 0.25*floor

        if base_exc is not None:
            axis_scale = np.array([1.0,0.6,0.9])
            accel_modes = [ResonantMode(f0=m['f0'], zeta=m['zeta'], gain=m['gain'], axes=tuple(m['axes'])) for m in vib_cfg['accel_modes']]
            gyro_modes = [ResonantMode(f0=m['f0'], zeta=m['zeta'], gain=m['gain'], axes=tuple(m['axes'])) for m in vib_cfg['gyro_modes']]
            for ax in range(3):
                vib_accel[:,ax] = axis_scale[ax] * apply_filter_bank(base_exc, fs, [m for m in accel_modes if ax in m.axes])
                vib_gyro[:,ax]  = axis_scale[ax] * apply_filter_bank(base_exc, fs, [m for m in gyro_modes if ax in m.axes])
            gyro_coupling = vib_cfg['g_sensitivity'] * vib_accel

    # Quantization LSB
    qa = (2 * cfg['imu']['accel_fs_g'] * 9.80665) / (2**cfg['imu']['quantization_bits'])
    qg = (2 * np.deg2rad(cfg['imu']['gyro_fs_dps'])) / (2**cfg['imu']['quantization_bits'])
    g_W = np.array([0, 0, 9.80665])
    f_meas = np.zeros((N,3)); w_meas = np.zeros((N,3))
    for k in range(N):
        R = R_WB[k]; w = w_B[k]; aw = a_W[k]; wdot = wdot_B[k]
        a_cent = np.cross(w, np.cross(w, r_lever))
        a_eul  = np.cross(wdot, r_lever)
        f_B = R.T @ (aw - g_W) + a_cent + a_eul
        ba = gm_a.step(dt); bg = gm_g.step(dt)
        f = (I + M_a) @ (S_a * f_B) + vib_accel[k] + ba + (na/np.sqrt(dt))*np.random.randn(3)
        ww = (I + M_g) @ (S_g * w) + vib_gyro[k] + gyro_coupling[k] + bg + (ng/np.sqrt(dt))*np.random.randn(3)
        f = np.clip(f, -cfg['imu']['accel_fs_g']*9.80665, cfg['imu']['accel_fs_g']*9.80665)
        ww = np.clip(ww, -np.deg2rad(cfg['imu']['gyro_fs_dps']), np.deg2rad(cfg['imu']['gyro_fs_dps']))
        f_meas[k] = quantize(f, qa)
        w_meas[k] = quantize(ww, qg)
    params: Dict[str, Any] = {
        'seed': seed,
        'dt': float(dt),
        'lever_arm_m': r_lever.tolist(),
        'misalignment_pct_draw': float(mis_pct*100.0),
        'scale_a': float(S_a),
        'scale_g': float(S_g),
        'M_a': M_a.tolist(),
        'M_g': M_g.tolist(),
        'accel': {
            'bias_init': b_a0.tolist(),
            'bias_tau_s': float(tau_a),
            'noise_density': float(na),
        },
        'gyro': {
            'bias_init': b_g0.tolist(),
            'bias_tau_s': float(tau_g),
            'noise_density': float(ng),
        },
        'quantization_bits': int(cfg['imu']['quantization_bits']),
        'accel_fs_g': float(cfg['imu']['accel_fs_g']),
        'gyro_fs_dps': float(cfg['imu']['gyro_fs_dps']),
        'vibration_used': vib_cfg is not None and (rpm_profile is not None or base_excitation_signal is not None),
        'vibration_cfg': vib_cfg if vib_cfg is not None else None,
    }
    return {'f_meas': f_meas, 'w_meas': w_meas}, params
