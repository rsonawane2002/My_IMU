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

# ============================================================================
# CONFIG
# ============================================================================

ROBOT_PATH = "/World/franka"
IMU_PATH = "/World/franka/panda_hand/Imu_Sensor"
OUTPUT_DIR = os.path.expanduser("~/Documents/trajectories")

# ============================================================================
# STARTING TRAJECTORY INDEX - MODIFY THIS VALUE
# ============================================================================
# Default: 1
# If you want to start from trajectory 200, change this to 200
# Folders will be named traj_1, traj_2, etc. (or traj_200, traj_201, etc.)
STARTING_TRAJ_INDEX = 205
# ============================================================================

NUM_TRAJECTORIES = 50
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
# TRAJECTORY SMOOTHING - THE KEY FIX
# ============================================================================

def smooth_interp(t):
    """
    Quintic polynomial: smooth position, velocity, AND acceleration
    This is the main fix for jerky motion!
    """
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
        
        imu = IMUSensor(prim_path=IMU_PATH, name="imu")
        
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
print("  - imu_X.csv - NVIDIA IMU sensor data")
print("  - clean_X.csv - Physics-computed IMU data")
print("Press PLAY to begin.")