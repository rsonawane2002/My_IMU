import numpy as np, yaml
from imu_sim2real_plus.sensors.imu_synth import synth_measurements
def test_synth_runs():
    cfg = yaml.safe_load(open('imu_sim2real_plus/config/example_config.yaml','r'))
    N=100; R=np.repeat(np.eye(3)[None,...], N, axis=0)
    w=np.zeros((N,3)); aW=np.zeros((N,3)); wdot=np.zeros((N,3))
    r=np.array(cfg['mount']['lever_arm_m']); dt=1/400
    out = synth_measurements(R,w,aW,wdot,r,cfg,dt,seed=0)
    assert out['f_meas'].shape == (N,3)
    assert out['w_meas'].shape == (N,3)
