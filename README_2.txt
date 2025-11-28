IMU Vibration Noise Simulator
=============================

What this models
----------------
- RPM-locked excitation (motor/gear/drive) -> passes through 2nd-order resonant modes.
- Multiple resonant modes per axis with damping and gains.
- Broad colored vibration floor (ARMA(1,1)).
- Gyro g-sensitivity (linear coupling from accel vibration).
- Sensor noise (white) and gyro bias random walk.
- Optional quantization (accel in mg LSB, gyro in dps LSB).

How to use
----------
1. Import the module:
   >>> from imu_vibration_sim import ResonantMode, simulate_imu_with_vibration

2. Define time base, true motion, RPM profile, and modes:
   >>> t = np.arange(0, 60, 1/400)
   >>> true_w = np.zeros((len(t),3)); true_a = np.zeros((len(t),3)); rpm = 800+100*np.sin(2*np.pi*0.02*t)
   >>> modes_accel = [ResonantMode(55,0.03,3.0,(0,2)), ResonantMode(120,0.02,2.0,(0,1,2))]
   >>> modes_gyro  = [ResonantMode(55,0.035,0.8,(0,2)), ResonantMode(180,0.02,1.4,(0,1))]

3. Simulate:
   >>> w_meas, a_meas, aux = simulate_imu_with_vibration(t, true_w, true_a, rpm, modes_accel, modes_gyro)

Tips
----
- Set f0, zeta, gain from your measured PSD peaks: f0 at peak, zeta from bandwidth BW ≈ 2*zeta*f0.
- Drive harmonics in synth_motor_excitation() to match your gear mesh (e.g., teeth * RPM/60).
- To emulate aliasing issues, raise a mode above Nyquist or lower fs.
- To emulate joint/end-effector differences, use different mode lists per mounting.
