"""
Keyboard Controls:
    - W/S: Increase/Decrease camera pitch (look up/down)
    - A/D: Decrease/Increase camera yaw (pan left/right)
    - UP/DOWN ARROW: Zoom in/out (decrease/increase camera distance)
    - 1: Switch to Circle formation
    - 2: Switch to V-shape formation
    - 3: Switch to Line formation
    - 4: Switch to Square formation

Example
-------
In a terminal, run as:
    $ python multidrone_formations.py

"""

import os
import time
import argparse
import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# ------------------------ DEFAULT PARAMETERS -------------------------
DEFAULT_DRONE = DroneModel.CF2X
DEFAULT_NUM_DRONES = 4
DEFAULT_PHYSICS = Physics.PYB
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Formation parameters
FORMATION_RADIUS = 0.5  # For circle and square formations
FORMATION_SPACING = 0.3  # For line and V-shape formations
BASE_ALTITUDE = 0.1
ALTITUDE_STEP = 0.05
ROTATION_SPEED = 0.3  # radians per second, for rotating formations

# Formation types
FORMATION_CIRCLE = 0
FORMATION_V = 1
FORMATION_LINE = 2
FORMATION_SQUARE = 3

def run(drone=DEFAULT_DRONE,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB):

    ############################################################################
    # 1. INITIALIZE ENVIRONMENT AND DRONE POSITIONS
    ############################################################################
    # Start with circle formation
    init_xyzs = get_formation_positions(FORMATION_CIRCLE, num_drones, 0)
    init_rpys = np.array([[0, 0, 0] for _ in range(num_drones)])

    # Create environment
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )
    PYB_CLIENT = env.getPyBulletClient()

    ############################################################################
    # 2. INITIALIZE LOGGING
    ############################################################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab
    )

    ############################################################################
    # 3. INITIALIZE DRONE CONTROLLERS AND ACTIONS
    ############################################################################
    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
    # Reduce the max velocity by setting a lower limit
    for c in ctrl:
        c.P_COEFF_FOR = np.array([0.2, 0.2, 1.2])  # original: [0.4, 0.4, 1.25]
        
        c.D_COEFF_FOR = np.array([0.15, 0.15, 0.45])  # original: [0.2, 0.2, 0.5]

    action = np.zeros((num_drones, 4))

    ############################################################################
    # 4. INITIALIZE CAMERA PARAMETERS FOR DYNAMIC VIEW
    ############################################################################
    camera_distance = 3.0
    camera_yaw = 50    # degrees
    camera_pitch = -35 # degrees
    camera_target = [0, 0, 0]

    ############################################################################
    # 5. INITIALIZE FORMATION PARAMETERS
    ############################################################################
    current_formation = FORMATION_CIRCLE
    formation_text_id = None  # For displaying the current formation name

    # Helper function to update the formation text display
    def update_formation_text():
        nonlocal formation_text_id
        formation_names = {
            FORMATION_CIRCLE: "Circle Formation",
            FORMATION_V: "V-Shape Formation",
            FORMATION_LINE: "Line Formation",
            FORMATION_SQUARE: "Square Formation"
        }
        if formation_text_id is not None:
            p.removeUserDebugItem(formation_text_id)
        formation_text_id = p.addUserDebugText(
            formation_names[current_formation],
            [0, 0, 1.5],
            textColorRGB=[1, 1, 0],
            textSize=1.5
        )

    # Initial text display
    if gui:
        update_formation_text()

    ############################################################################
    # 6. RUN THE SIMULATION LOOP
    ############################################################################
    START = time.time()
    for t_step in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        sim_time = t_step / env.CTRL_FREQ

        # Get target positions based on current formation and time
        target_positions = get_formation_positions(current_formation, num_drones, sim_time)

        # Compute control for each drone
        for i in range(num_drones):
            x_des, y_des, z_des = target_positions[i]

            action[i, :], _, _ = ctrl[i].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[i],
                target_pos=np.array([x_des, y_des, z_des]),
                target_rpy=np.array([0, 0, 0])
            )

            # Log with a 12-element control vector (pad with zeros)
            control_vector = np.array([
                x_des, y_des, z_des,  # Desired position (3)
                0, 0, 0,             # Desired orientation placeholders (3)
                0, 0, 0, 0, 0, 0      # Extra placeholders to reach 12 elements (6)
            ])
            logger.log(
                drone=i,
                timestamp=sim_time,
                state=obs[i],
                control=control_vector
            )

        # Render the environment
        env.render()

        # ---------------- Dynamic Camera Control ----------------
        # Read keyboard events from PyBullet
        keys = p.getKeyboardEvents()

        # Adjust pitch: W (increase pitch upward), S (decrease pitch, look down)
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            camera_pitch -= 1
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            camera_pitch += 1

        # A (pan left), D (pan right)
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            camera_yaw -= 1
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            camera_yaw += 1

        # Adjust zoom (distance): UP_ARROW (zoom in), DOWN_ARROW (zoom out)
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            camera_distance = max(0.5, camera_distance - 0.2)
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            camera_distance += 0.1

        # Formation switching keys (1-4)
        formation_changed = False
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_RELEASED:
            current_formation = FORMATION_CIRCLE
            formation_changed = True
        elif ord('2') in keys and keys[ord('2')] & p.KEY_WAS_RELEASED:
            current_formation = FORMATION_V
            formation_changed = True
        elif ord('3') in keys and keys[ord('3')] & p.KEY_WAS_RELEASED:
            current_formation = FORMATION_LINE
            formation_changed = True
        elif ord('4') in keys and keys[ord('4')] & p.KEY_WAS_RELEASED:
            current_formation = FORMATION_SQUARE
            formation_changed = True
            
        # Update formation text if changed
        if formation_changed and gui:
            update_formation_text()

        camera_target = [0, 0, 0]  # Center of the scene

        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)

        if gui:
            sync(t_step, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("multi_formations")
    if plot:
        logger.plot()

def get_formation_positions(formation_type, num_drones, sim_time):
    """
    Calculate drone positions for different formations.
    
    Args:
        formation_type (int): Type of formation (CIRCLE, V, LINE, SQUARE)
        num_drones (int): Number of drones
        sim_time (float): Current simulation time
        
    Returns:
        numpy.ndarray: Array of positions [x, y, z] for each drone
    """
    positions = []
    
    if formation_type == FORMATION_CIRCLE:
        # Rotating circle formation
        for i in range(num_drones):
            angle = 2 * math.pi * i / num_drones + ROTATION_SPEED * sim_time
            x = FORMATION_RADIUS * math.cos(angle)
            y = FORMATION_RADIUS * math.sin(angle)
            z = BASE_ALTITUDE + i * ALTITUDE_STEP
            positions.append([x, y, z])
            
    elif formation_type == FORMATION_V:
        # V-shape formation
        if num_drones == 1:
            positions.append([0, 0, BASE_ALTITUDE])
        else:
            mid_point = (num_drones - 1) / 2
            for i in range(num_drones):
                # Calculate position in V-shape
                x_offset = abs(i - mid_point) * FORMATION_SPACING
                y_offset = -(i - mid_point) * FORMATION_SPACING
                
                # Add some rotation for dynamic movement
                if sim_time > 0:
                    rotation_angle = ROTATION_SPEED * 0.5 * sim_time
                    x = x_offset * math.cos(rotation_angle) - y_offset * math.sin(rotation_angle)
                    y = x_offset * math.sin(rotation_angle) + y_offset * math.cos(rotation_angle)
                else:
                    x = x_offset
                    y = y_offset
                    
                z = BASE_ALTITUDE + i * ALTITUDE_STEP
                positions.append([x, y, z])
                
    elif formation_type == FORMATION_LINE:
        # Line formation
        for i in range(num_drones):
            # Spread drones evenly in a line
            offset = (i - (num_drones - 1) / 2) * FORMATION_SPACING
            
            # Adds some movement
            y_wave = 0.1 * math.sin(sim_time + i * 0.5)
            
            x = offset
            y = y_wave
            z = BASE_ALTITUDE + i * ALTITUDE_STEP
            positions.append([x, y, z])
            
    elif formation_type == FORMATION_SQUARE:
        # Square formation
        if num_drones <= 4:
            # With 4 or fewer drones, place them at the corners
            corners = [
                [FORMATION_RADIUS, FORMATION_RADIUS],
                [FORMATION_RADIUS, -FORMATION_RADIUS],
                [-FORMATION_RADIUS, -FORMATION_RADIUS],
                [-FORMATION_RADIUS, FORMATION_RADIUS]
            ]
            
            for i in range(num_drones):
                x, y = corners[i]
                z = BASE_ALTITUDE + i * ALTITUDE_STEP
                positions.append([x, y, z])
        else:
            # With more drones, distribute them evenly around the perimeter
            perimeter = 8 * FORMATION_RADIUS  
            segment_length = perimeter / num_drones 
            
            for i in range(num_drones):
                position = i * segment_length
                
                # Determine which side of the square
                side = position // (2 * FORMATION_RADIUS)
                remainder = position % (2 * FORMATION_RADIUS)
                
                if side == 0:  # Bottom edge
                    x = -FORMATION_RADIUS + remainder
                    y = -FORMATION_RADIUS
                elif side == 1:  # Right edge
                    x = FORMATION_RADIUS
                    y = -FORMATION_RADIUS + remainder
                elif side == 2:  # Top edge
                    x = FORMATION_RADIUS - remainder
                    y = FORMATION_RADIUS
                else:  # Left edge
                    x = -FORMATION_RADIUS
                    y = FORMATION_RADIUS - remainder
                
                # Add rotation for dynamic movement
                if sim_time > 0:
                    rotation_angle = ROTATION_SPEED * 0.3 * sim_time
                    x_rot = x * math.cos(rotation_angle) - y * math.sin(rotation_angle)
                    y_rot = x * math.sin(rotation_angle) + y * math.cos(rotation_angle)
                    x, y = x_rot, y_rot
                
                z = BASE_ALTITUDE + i * ALTITUDE_STEP
                positions.append([x, y, z])
    
    return np.array(positions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-drone formations with dynamic camera demo")
    parser.add_argument("--drone", default=DEFAULT_DRONE, type=DroneModel, help="Drone model", choices=DroneModel)
    parser.add_argument("--num_drones", default=DEFAULT_NUM_DRONES, type=int, help="Number of drones")
    parser.add_argument("--physics", default=DEFAULT_PHYSICS, type=Physics, help="Physics updates", choices=Physics)
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, help="Use PyBullet GUI")
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VISION, type=str2bool, help="Record video")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, help="Plot simulation results")
    parser.add_argument("--user_debug_gui", default=DEFAULT_USER_DEBUG_GUI, type=str2bool, help="User debug GUI")
    parser.add_argument("--obstacles", default=DEFAULT_OBSTACLES, type=str2bool, help="Add obstacles to environment")
    parser.add_argument("--simulation_freq_hz", default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help="Simulation frequency")
    parser.add_argument("--control_freq_hz", default=DEFAULT_CONTROL_FREQ_HZ, type=int, help="Control frequency")
    parser.add_argument("--duration_sec", default=DEFAULT_DURATION_SEC, type=int, help="Duration of the simulation")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, help="Folder where logs are saved")
    parser.add_argument("--colab", default=DEFAULT_COLAB, type=str2bool, help="Are we running in a notebook/Colab?")
    ARGS = parser.parse_args()

    run(**vars(ARGS))