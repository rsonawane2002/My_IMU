import omni.timeline
import omni.usd
import csv
import os
import sys
import math
import numpy as np
import random
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.kit.app import get_app
from omni.isaac.sensor import IMUSensor

# --- C++ Module Import ---

# change path to point to .so file
compiled_code_path = "/home/sameer/Documents/imu_sim2real_plus_v2/sensor_setup/"

if compiled_code_path not in sys.path:
    sys.path.append(compiled_code_path)

try:
    import sim2real_native
    print("SUCCESS: Loaded C++ Noise Engine from " + compiled_code_path)
except ImportError as e:
    print("CRITICAL ERROR: Could not load C++ extension.")
    print("Specific Error: " + str(e))

# --- Wrapper Class ---
class Sim2RealIMUSensor:
    def __init__(self, prim_path, name="imu", seed=123):
        self._sensor = IMUSensor(prim_path=prim_path, name=name)
        self._timeline = omni.timeline.get_timeline_interface()
        self._core = sim2real_native.Sim2RealCore(seed)

    def initialize(self, physics_sim_view=None):
        self._sensor.initialize(physics_sim_view)

    def get_current_frame(self, read_gravity=True):
        raw = self._sensor.get_current_frame(read_gravity=read_gravity)
        current_time = self._timeline.get_current_time()
        
        # Pass to C++ engine
        noisy_data = self._core.process(raw['lin_acc'], raw['ang_vel'], current_time)
        
        raw['lin_acc'] = noisy_data['lin_acc']
        raw['ang_vel'] = noisy_data['ang_vel']
        return raw

    def __getattr__(self, name):
        return getattr(self._sensor, name)

# --- Config ---
ROBOT_PATH = "/World/franka"
IMU_PATH = "/World/franka/panda_hand/Imu_Sensor"
OUTPUT_DIR = os.path.expanduser("~/Documents/trajectories")

STARTING_TRAJ_INDEX = 251
NUM_TRAJECTORIES = 500
TRAJ_DURATION = 6.0
SETTLE_TIME = 2.0
RESET_TIME = 3.0
STARTUP_FRAMES = 60

HOME = np.array([0.000167, -0.786, 4.01e-5, -2.35502, 9.29e-5, 1.571, 0.786, 0.04, 0.04])

def smooth_interp(t):
    return 10*t**3 - 15*t**4 + 6*t**5

# --- State ---
robot = None
imu = None
timeline = omni.timeline.get_timeline_interface()

frame = 0
phase = 'startup'
traj_idx = STARTING_TRAJ_INDEX
time_in_phase = 0.0
targets = []

imu_isaac_file = None
imu_isaac_writer = None

# --- Helpers ---
def gen_target():
    t = HOME.copy()
    MAX_OFFSET = math.pi / 5
    for i in range(7):
        offset = random.uniform(-MAX_OFFSET, MAX_OFFSET)
        t[i] = HOME[i] + offset
    return t

def start_log():
    global imu_isaac_file, imu_isaac_writer
    
    traj_folder = os.path.join(OUTPUT_DIR, "traj_" + str(traj_idx))
    os.makedirs(traj_folder, exist_ok=True)
    
    imu_isaac_path = os.path.join(traj_folder, "imu_" + str(traj_idx) + ".csv")
    imu_isaac_file = open(imu_isaac_path, 'w', newline='')
    imu_isaac_writer = csv.writer(imu_isaac_file)
    imu_isaac_writer.writerow(['time', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 
                               'gripper1', 'gripper2', 'imu_ax', 'imu_ay', 'imu_az',
                               'imu_gx', 'imu_gy', 'imu_gz'])

def log(pos, dt):
    global imu_isaac_writer, imu_isaac_file
    if imu and imu_isaac_writer:
        current_time = timeline.get_current_time()
        imu_data = imu.get_current_frame(read_gravity=True)
        acc = imu_data['lin_acc']
        vel = imu_data['ang_vel']
        
        imu_isaac_writer.writerow([current_time] + list(pos[:9]) + 
                                  [acc[2], -1 * acc[1], acc[0], vel[2], -1 * vel[1], vel[0]])
        imu_isaac_file.flush()

def close_log():
    global imu_isaac_file, imu_isaac_writer
    if imu_isaac_file: imu_isaac_file.close()
    imu_isaac_file = imu_isaac_writer = None

# --- Main Loop ---
def update(e):
    global robot, imu, frame, phase, traj_idx, time_in_phase, targets
    
    if not timeline.is_playing(): return
    
    frame += 1
    dt = e.payload["dt"]
    
    if robot is None:
        robot = Franka(prim_path=ROBOT_PATH, name="franka")
        robot.initialize()
        robot.set_joints_default_state(positions=HOME, velocities=np.zeros(9))
        robot.post_reset()
        
        imu = Sim2RealIMUSensor(prim_path=IMU_PATH, name="imu")
        targets = [gen_target() for _ in range(NUM_TRAJECTORIES)]
        return
    
    if phase == 'startup':
        if frame > STARTUP_FRAMES:
            phase = 'moving'
            time_in_phase = 0.0
            start_log()
            print("Trajectory " + str(traj_idx))
        return
    
    time_in_phase += dt
    pos = robot.get_joint_positions()
    
    if phase == 'moving':
        progress = min(time_in_phase / TRAJ_DURATION, 1.0)
        t = smooth_interp(progress)
        target_pos = HOME + t * (targets[traj_idx - STARTING_TRAJ_INDEX] - HOME)
        robot.apply_action(ArticulationAction(joint_positions=target_pos))
        log(pos, dt)
        if progress >= 1.0:
            phase = 'settling'
            time_in_phase = 0.0
    
    elif phase == 'settling':
        log(pos, dt)
        if time_in_phase >= SETTLE_TIME:
            close_log()
            traj_idx += 1
            if traj_idx >= STARTING_TRAJ_INDEX + NUM_TRAJECTORIES:
                print("Done: All trajectories collected")
                timeline.stop()
                return
            targets.append(gen_target())
            phase = 'resetting'
            time_in_phase = 0.0
    
    elif phase == 'resetting':
        progress = min(time_in_phase / RESET_TIME, 1.0)
        t = smooth_interp(progress)
        reset_pos = pos + t * (HOME - pos)
        robot.apply_action(ArticulationAction(joint_positions=reset_pos))
        if progress >= 1.0:
            robot.set_joint_positions(HOME)
            robot.set_joint_velocities(np.zeros(9)) 
            phase = 'moving'
            time_in_phase = 0.0
            start_log()
            print("Trajectory " + str(traj_idx))

# --- Start ---
stream = get_app().get_update_event_stream()
sub = stream.create_subscription_to_pop(update)
print("Collecting " + str(NUM_TRAJECTORIES) + " trajectories -> " + OUTPUT_DIR)
print("Press PLAY to begin.")