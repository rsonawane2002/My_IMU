# imu_sim2real_plus (Isaac-ready scaffold)

A modular scaffold to synthesize ASM330LHH-class IMU data, validate realism with Allan deviation, compute KPIs, and wire to Isaac Sim/ROS 2.

## Quick start
```bash
pip install -r requirements.txt
python -m imu_sim2real_plus.scripts.generate_dataset --config imu_sim2real_plus/config/example_config.yaml --out runs/sim --sequences 3 --seconds 10 --odr 400
python -m imu_sim2real_plus.scripts.run_allan --series runs/sim/seq_00000_f_meas.npy --fs 400
```
## Isaac wiring
Provide a Python callback in Isaac returning R_WB(3x3), w_B(3,), a_W(3,), wdot_B(3,) at ODR; then call the synth pipeline.
