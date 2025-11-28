import numpy as np
from imu_sim2real_plus.metrics.allan import allan_deviation
def test_allan_runs():
    x = np.random.randn(10000)
    taus = np.array([0.1,0.2,0.5,1.0])
    s = allan_deviation(x, fs=100.0, taus=taus)
    assert s.shape == taus.shape
