import math
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TimestampCfg:
    odr_hz: int = 400
    clock_drift_ppm: Tuple[float, float] = (-30.0, 30.0)
    jitter_us_rms: float = 150.0
    packet_burst: int = 10
    drop_prob: float = 0.01

@dataclass
class AxisNoiseCfg:
    noise_density: Tuple[float, float]
    scale_ppm: Tuple[float, float]
    bias_init: Tuple[float, float]
    bias_tau_s: Tuple[float, float]

@dataclass
class IMUCfg:
    accel_fs_g: int = 8
    gyro_fs_dps: int = 2000
    accel: AxisNoiseCfg = field(default_factory=lambda: AxisNoiseCfg(
        noise_density=(60e-6*9.80665, 120e-6*9.80665),
        scale_ppm=(-3000, 3000),
        bias_init=(-15e-3*9.80665, 15e-3*9.80665),
        bias_tau_s=(200, 2000)))
    gyro: AxisNoiseCfg = field(default_factory=lambda: AxisNoiseCfg(
        noise_density=(6e-5, 8e-5),  # rad/s/√Hz (approx mdps/√Hz range)
        scale_ppm=(-1000, 1000),
        bias_init=(-1.94e-5, 1.94e-5),  # rad/s (~±4 °/h)
        bias_tau_s=(200, 2000)))
    misalignment_pct: Tuple[float, float] = (-1.0, 1.0)
    quantization_bits: int = 16
    timestamp: TimestampCfg = field(default_factory=TimestampCfg)

@dataclass
class MountCfg:
    lever_arm_m: Tuple[float, float, float] = (0.0, 0.0, 0.05)
    misrot_deg: Tuple[float, float] = (-20.0, 20.0)

@dataclass
class DatasetCfg:
    out_dir: str = "runs/sim"
    num_sequences: int = 100
    seconds_per_seq: float = 30.0
    random_seed: int = 1234

@dataclass
class KPIThresholds:
    orient_med_deg: float = 1.5
    orient_p95_deg: float = 3.0

@dataclass
class Config:
    imu: IMUCfg = field(default_factory=IMUCfg)
    mount: MountCfg = field(default_factory=MountCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    kpi: KPIThresholds = field(default_factory=KPIThresholds)
