import omni.timeline
import omni.usd
import csv
import os
from datetime import datetime
import math
import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.kit.app import get_app
from omni.isaac.sensor import IMUSensor
import random
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ============================================================================
# NEW CLASSES FOR NOISE LOGIC (From imu_vibration_sim)
# ============================================================================

@dataclass
class ResonantMode:
    f0: float
    zeta: float
    gain: float
    axes: Tuple[int, ...] = (0, 1, 2)

class StatefulFilter:
    """Helper to run scipy.signal.lfilter frame-by-frame using 'zi' state."""
    def __init__(self, b, a):
        self.b = b
        self.a = a
        # Initial filter state (zeros)
        self.zi = signal.lfilter_zi(b, a) * 0.0

    def step(self, x):
        # Filter a single sample 'x' and update state 'zi'
        y, self.zi = signal.lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]

def bilinear_resonator(fs: float, mode: ResonantMode):
    """Calculates IIR coefficients for a resonant mode."""
    wn = 2 * np.pi * mode.f0
    zeta = mode.zeta
    num = [0, 2*zeta*wn, 0]
    den = [1, 2*zeta*wn, wn**2]
    num = [c*mode.gain for c in num]
    bz, az = signal.bilinear(num, den, fs=fs)
    return np.array(bz, dtype=float), np.array(az, dtype=float)

# ============================================================================
# MODIFIED CUSTOM SENSOR CLASS
# ============================================================================

class Sim2RealIMUSensor:
    def __init__(self, prim_path, name="imu", seed=123):
        # Keep original sensor initialization
        self._sensor = IMUSensor(prim_path=prim_path, name=name)
        self._timeline = omni.timeline.get_timeline_interface()
        
        # --- Noise Injection State Initialization ---
        self.rng = np.random.default_rng(seed)
        self.filters_initialized = False
        self.last_time = None
        
        # 1. Motor State
        self.motor_phase = 0.0
        
        # 2. ARMA Noise State (previous outputs)
        self.arma_prev_y = 0.0
        self.arma_prev_e = 0.0
        
        # 3. Bias Random Walk State
        self.gyro_bias = np.zeros(3)
        
        # 4. Filters (Dictionaries to hold StatefulFilter instances)
        self.filters_acc = {} 
        self.filters_gyr = {}
        
        # Config (Matched to imu_vibration_sim logic)
        self.modes_acc = [
            ResonantMode(f0=75.0,  zeta=0.02, gain=0.05, axes=(0,1,2)),
            ResonantMode(f0=95.0,  zeta=0.03, gain=0.02, axes=(0,1,2)),
            ResonantMode(f0=120.0, zeta=0.03, gain=0.04, axes=(2,)),  
            ResonantMode(f0=130.0, zeta=0.03, gain=0.03, axes=(2,)),
        ]
        self.modes_gyr = [
            ResonantMode(f0=6.5,   zeta=0.07, gain=0.02, axes=(0,1,2)),
            ResonantMode(f0=8.0,   zeta=0.08, gain=0.015, axes=(0,1,2)),
            ResonantMode(f0=75.0,  zeta=0.03, gain=0.004, axes=(2,)),
            ResonantMode(f0=120.0, zeta=0.03, gain=0.004, axes=(2,)),
        ]
        

        #default config
        #self.accel_white_std = 0.02
        self.accel_white_std = 0.2

        self.gyro_white_std = 0.0025
        self.gyro_bias_rw_std = 2e-5
        self.g_sensitivity = 0.002
        self.axis_scale = np.array([1.0, 0.6, 0.9])

    def initialize(self, physics_sim_view=None):
        self._sensor.initialize(physics_sim_view)

    def _init_filters(self, dt):
        """Initialize filter coefficients once we know the timestep (fs)."""
        if dt <= 0: return
        fs = 1.0 / dt
        
        for i, mode in enumerate(self.modes_acc):
            b, a = bilinear_resonator(fs, mode)
            self.filters_acc[i] = StatefulFilter(b, a)
            
        for i, mode in enumerate(self.modes_gyr):
            b, a = bilinear_resonator(fs, mode)
            self.filters_gyr[i] = StatefulFilter(b, a)
            
        self.filters_initialized = True

    def get_current_frame(self, read_gravity=True):
        # 1. Get Base Data from NVIDIA Sensor
        raw = self._sensor.get_current_frame(read_gravity=read_gravity)
        
        # Extract clean vectors
        true_acc = raw['lin_acc']
        true_ang = raw['ang_vel']

        # 2. Calculate dt (frame-by-frame)
        current_time = self._timeline.get_current_time()
        if self.last_time is None:
            dt = 0.0 # First frame, skip noise gen to establish baseline
        else:
            dt = current_time - self.last_time
        self.last_time = current_time

        # If dt is invalid or too small, just return raw data
        if dt <= 1e-6:
            return raw

        # 3. Initialize filters if this is the first valid frame
        if not self.filters_initialized:
            self._init_filters(dt)

        # =========================================================
        # INJECT NOISE (Adapted from simulate_imu_with_vibration)
        # =========================================================
        
        # A. Motor Excitation (Harmonics)
        rpm = 4500.0 * 60.0 # Adapted from 75Hz * 60 = 4500 RPM
        self.motor_phase += 2 * np.pi * (rpm / 60.0) * dt
        
        exc = 0.0
        harmonics = {1:1.0, 2:0.4, 3:0.2} # from synth_motor_excitation
        for h, amp in harmonics.items():
            exc += amp * np.sin(h * self.motor_phase) # Ignoring phase noise for frame-consistency

        # B. ARMA Colored Noise (Floor)
        sigma, ar, ma = 0.4, 0.96, 0.2
        current_e = self.rng.standard_normal() * sigma
        floor = ar * self.arma_prev_y + current_e + ma * self.arma_prev_e
        
        # Update ARMA state
        self.arma_prev_y = floor
        self.arma_prev_e = current_e

        # C. Base Excitation
        base_exc = exc + 0.25 * floor

        # D. Apply Filter Banks (Vibration)
        vib_accel = np.zeros(3)
        vib_gyro = np.zeros(3)

        if self.filters_initialized:
            # Accumulate Accel Vibration
            for i, mode in enumerate(self.modes_acc):
                val = self.filters_acc[i].step(base_exc)
                for ax in mode.axes:
                    vib_accel[ax] += self.axis_scale[ax] * val
            
            # Accumulate Gyro Vibration
            for i, mode in enumerate(self.modes_gyr):
                val = self.filters_gyr[i].step(base_exc)
                for ax in mode.axes:
                    vib_gyro[ax] += self.axis_scale[ax] * val

        # E. Coupling & White Noise
        gyro_coupling = self.g_sensitivity * vib_accel
        n_acc = self.rng.normal(0, self.accel_white_std, size=3)
        n_gyr = self.rng.normal(0, self.gyro_white_std, size=3)

        # F. Gyro Bias Random Walk
        bias_step = self.rng.normal(0, self.gyro_bias_rw_std * np.sqrt(dt), size=3)
        self.gyro_bias += bias_step

        # G. Final Summation (NO GRAVITY ADDED/SUBTRACTED)
        # Note: We modify the 'raw' dict in place or create copy
        
        final_acc = true_acc + vib_accel + n_acc
        final_ang = true_ang + vib_gyro + gyro_coupling + n_gyr + self.gyro_bias

        raw['lin_acc'] = final_acc
        raw['ang_vel'] = final_ang

        return raw

    def __getattr__(self, name):
        return getattr(self._sensor, name)

# ============================================================================
# CONFIG
# ============================================================================

ROBOT_PATH = "/World/franka"
IMU_PATH = "/World/franka/panda_hand/Imu_Sensor"
OUTPUT_DIR = os.path.expanduser("~/Documents/trajectories")

# ============================================================================
# STARTING TRAJECTORY INDEX - MODIFY THIS VALUE
# ============================================================================
STARTING_TRAJ_INDEX = 251
# ============================================================================

NUM_TRAJECTORIES = 500
TRAJ_DURATION = 6.0
SETTLE_TIME = 2.0
RESET_TIME = 3.0
STARTUP_FRAMES = 60

HOME = np.array([0.000167, -0.786, 4.01e-5, -2.35502, 9.29e-5, 1.571, 0.786, 0.04, 0.04])

# Slightly more conservative limits to reduce collision risk
LIMITS = [
    (-1.3, 1.3),   # J1
    (-0.9, 0.9),   # J2
    (-1.3, 1.3),   # J3
    (-2.0, -0.5),  # J4 - Keep away from full extension
    (-1.5, 1.5),   # J5
    (1.2, 2.3),    # J6
    (-1.3, 1.3),   # J7
    (0.01, 0.035), # Gripper 1
    (0.01, 0.035)  # Gripper 2
]


# ============================================================================
# TRAJECTORY SMOOTHING
# ============================================================================

def smooth_interp(t):
    return 10*t**3 - 15*t**4 + 6*t**5

# ============================================================================
# STATE
# ============================================================================

robot = None
imu = None
kinematics_solver = None
rtb_robot = None
timeline = omni.timeline.get_timeline_interface()

frame = 0
phase = 'startup'
traj_idx = STARTING_TRAJ_INDEX
time_in_phase = 0.0

targets = []

# IMU logging files
imu_isaac_file = None
imu_isaac_writer = None
clean_isaac_file = None
clean_isaac_writer = None

# History for numerical differentiation
prev_ee_pos = None
prev_ee_vel = None
prev_time = None

# ============================================================================
# HELPERS
# ============================================================================

def gen_target():
    t = HOME.copy()
    MAX_OFFSET = math.pi / 5   # 1.5708 rad

    for i in range(7):  # only 7 arm joints
        offset = random.uniform(-MAX_OFFSET, MAX_OFFSET)
        candidate = HOME[i] + offset
        t[i] = candidate
    return t

def compute_imu_data(joint_pos, joint_vel, dt):
    """Compute linear accel and velocity given joint pos and vel and dt with physics"""
    global prev_ee_pos, prev_ee_vel, prev_time

    # Get the end effector pose (with API call)
    ee_pos, ee_rot = kinematics_solver.compute_end_effector_pose(position_only=False)

    # Convert ee_rot to quaternion
    quat = R.from_matrix(ee_rot).as_quat()  # Returns [x, y, z, w]
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  

    # Compute linear velocity of ee
    if prev_ee_pos is not None and prev_time is not None:
        actual_dt = 1e-6
        if dt > 0:
            actual_dt = dt

        # Simple derivative calculation of velocity after change in timestep
        ee_vel = (ee_pos - prev_ee_pos) / actual_dt
    else:
        # x,y,z = 0s
        ee_vel = np.zeros(3)
    
    # Compute linear acc of ee
    if prev_ee_vel is not None and prev_time is not None:
        actual_dt = 1e-6
        if dt > 0:
            actual_dt = dt

        # Simple derivative calculation of velocity after change in timestep
        ee_acc = (ee_vel - prev_ee_vel) / actual_dt
    else:
        ee_acc = np.zeros(3)
    
    # Compute angular velocity with jacobian
    # Use only first 7 joints (exclude gripper fingers)
    q = joint_pos[:7]
    qd = joint_vel[:7]
    
    # Compute Jacobian in base frame (6x7 matrix)
    # Returns [linear_velocity; angular_velocity] = J * joint_velocities
    J = rtb_robot.jacob0(q)
    
    # Extract angular velocity from Jacobian mapping
    # J @ qd gives [vx, vy, vz, wx, wy, wz]
    ee_twist = J @ qd
    ang_vel = ee_twist[3:]  # Extract angular velocity components
    
    # Update history
    prev_ee_pos = ee_pos.copy()
    prev_ee_vel = ee_vel.copy()
    prev_time = timeline.get_current_time()
    
    return {
        'lin_acc': ee_acc,
        'ang_vel': ang_vel,
        'quat': quat_wxyz
    }

def reset_imu_history():
    """Reset IMU history when starting a new trajectory"""
    global prev_ee_pos, prev_ee_vel, prev_time
    prev_ee_pos = None
    prev_ee_vel = None
    prev_time = None

def start_log():
    global imu_isaac_file, imu_isaac_writer, clean_isaac_file, clean_isaac_writer
    
    # Create subfolder with trajectory index
    traj_folder = os.path.join(OUTPUT_DIR, f"traj_{traj_idx}")
    os.makedirs(traj_folder, exist_ok=True)
    
    # Original trajectory file
    path = os.path.join(traj_folder, f"traj_{traj_idx}.csv")
    # IMU Isaac sensor data file
    imu_isaac_path = os.path.join(traj_folder, f"imu_{traj_idx}.csv")
    imu_isaac_file = open(imu_isaac_path, 'w', newline='')
    imu_isaac_writer = csv.writer(imu_isaac_file)
    imu_isaac_writer.writerow(['time', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 
                               'gripper1', 'gripper2', 'imu_ax', 'imu_ay', 'imu_az',
                               'imu_gx', 'imu_gy', 'imu_gz'])
    
    # Clean Isaac computed data file
    clean_isaac_path = os.path.join(traj_folder, f"clean_{traj_idx}.csv")
    clean_isaac_file = open(clean_isaac_path, 'w', newline='')
    clean_isaac_writer = csv.writer(clean_isaac_file)
    clean_isaac_writer.writerow(['time', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 
                                 'gripper1', 'gripper2', 'imu_ax', 'imu_ay', 'imu_az',
                                 'imu_gx', 'imu_gy', 'imu_gz',
                                 'quat_x', 'quat_y', 'quat_z', 'quat_w'])
    
    reset_imu_history()

def log(pos, dt):
    if imu:
        current_time = timeline.get_current_time()
        
        # Log original IMU sensor data
        # Note: Sim2RealIMUSensor.get_current_frame now applies the noise logic internally
        imu_data = imu.get_current_frame(read_gravity=True)
        acc = imu_data['lin_acc']
        vel = imu_data['ang_vel']
        
        # Log IMU Isaac sensor data to separate file
        if imu_isaac_writer:
            imu_isaac_writer.writerow([current_time] + list(pos[:9]) + 
                                      [acc[2], -1 * acc[1], acc[0], vel[2], -1 * vel[1], vel[0]])
            imu_isaac_file.flush()
        
        # Compute and log physics-based IMU data
        if clean_isaac_writer:
            joint_vel = robot.get_joint_velocities()
            computed_imu = compute_imu_data(pos, joint_vel, dt)
            acc = computed_imu['lin_acc']
            ang_vel = computed_imu['ang_vel']
            quat = computed_imu['quat']
            
            clean_isaac_writer.writerow([current_time] + list(pos[:9]) + 
                                        [acc[2], -1 * acc[1], acc[0],
                                         -1 * ang_vel[2], ang_vel[1], ang_vel[0],
                                         quat[2], quat[1], quat[0], quat[3]])  # quat as x,y,z,w
            clean_isaac_file.flush()


def close_log():
    global imu_isaac_file, imu_isaac_writer, clean_isaac_file, clean_isaac_writer
    if imu_isaac_file:
        imu_isaac_file.close()
        imu_isaac_file = imu_isaac_writer = None
    if clean_isaac_file:
        clean_isaac_file.close()
        clean_isaac_file = clean_isaac_writer = None

# ============================================================================
# MAIN LOOP
# ============================================================================

def update(e):
    global robot, imu, kinematics_solver, rtb_robot, frame, phase, traj_idx, time_in_phase, targets
    
    if not timeline.is_playing():
        return
    
    frame += 1
    dt = e.payload["dt"]
    
    # Initialize
    if robot is None:
        robot = Franka(prim_path=ROBOT_PATH, name="franka")
        robot.initialize()
        robot.set_joints_default_state(positions=HOME, velocities=np.zeros(9))
        robot.post_reset()
        
        #imu = IMUSensor(prim_path=IMU_PATH, name="imu")
        imu = Sim2RealIMUSensor(prim_path=IMU_PATH, name="imu")
        
        # Initialize kinematics solver
        kinematics_solver = KinematicsSolver(robot)

        # Initialize robotics toolbox for rtb calculation
        rtb_robot = rtb.models.DH.Panda()
        
        targets = [gen_target() for _ in range(NUM_TRAJECTORIES)]
        return
    
    # Startup delay
    if phase == 'startup':
        if frame > STARTUP_FRAMES:
            phase = 'moving'
            time_in_phase = 0.0
            start_log()
            print(f"Trajectory {traj_idx}/{STARTING_TRAJ_INDEX + NUM_TRAJECTORIES - 1}")
        return
    
    time_in_phase += dt
    pos = robot.get_joint_positions()
    
    # Execute trajectory
    if phase == 'moving':
        progress = min(time_in_phase / TRAJ_DURATION, 1.0)
        t = smooth_interp(progress)  # Quintic instead of cosine
        target_pos = HOME + t * (targets[traj_idx - STARTING_TRAJ_INDEX] - HOME)
        robot.apply_action(ArticulationAction(joint_positions=target_pos))
        log(pos, dt)
        
        if progress >= 1.0:
            phase = 'settling'
            time_in_phase = 0.0
    
    # Settle at target
    elif phase == 'settling':
        log(pos, dt)
        
        if time_in_phase >= SETTLE_TIME:
            close_log()
            traj_idx += 1
            
            if traj_idx >= STARTING_TRAJ_INDEX + NUM_TRAJECTORIES:
                print(f"Done: {NUM_TRAJECTORIES} trajectories collected")
                timeline.stop()
                return
            
            # Generate new target for next trajectory
            targets.append(gen_target())
            
            phase = 'resetting'
            time_in_phase = 0.0
    
    # Reset to home with same smooth trajectory
    elif phase == 'resetting':
        progress = min(time_in_phase / RESET_TIME, 1.0)
        t = smooth_interp(progress)  # Quintic here too!
        reset_pos = pos + t * (HOME - pos)
        robot.apply_action(ArticulationAction(joint_positions=reset_pos))
        
        if progress >= 1.0:
            robot.set_joint_positions(HOME)
            robot.set_joint_velocities(np.zeros(9)) 
            phase = 'moving'
            time_in_phase = 0.0
            start_log()
            print(f"Trajectory {traj_idx}/{STARTING_TRAJ_INDEX + NUM_TRAJECTORIES - 1}")

# ============================================================================
# START
# ============================================================================

stream = get_app().get_update_event_stream()
sub = stream.create_subscription_to_pop(update)

print(f"Collecting {NUM_TRAJECTORIES} Franka trajectories (indices {STARTING_TRAJ_INDEX} to {STARTING_TRAJ_INDEX + NUM_TRAJECTORIES - 1}) -> {OUTPUT_DIR}")
print("Each trajectory will be in its own folder: traj_X/")
print("Files generated per trajectory:")
print("  - imu_X.csv - NVIDIA IMU sensor data (Sim2Real Noise Injected)")
print("  - clean_X.csv - Physics-computed IMU data")
print("Press PLAY to begin.")