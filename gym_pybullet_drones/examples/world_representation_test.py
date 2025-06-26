"""Script demonstrating the world representation for multi-drone planning.

Example
-------
In a terminal, run as:

    $ python world_representation_test.py

"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.GridWorldMap import GridWorldMap, CellState
from gym_pybullet_drones.utils.WorldObjects import ObjectType, ObjectOfInterest, Obstacle
from gym_pybullet_drones.utils.CentralizedPlanner import CentralizedPlanner

# Environment parameters
ENV_WIDTH = 10.0  # meters
ENV_LENGTH = 10.0  # meters
CELL_SIZE = 0.5  # meters
HEIGHT_THRESHOLD = 1.0  # meters
FOV_RADIUS = 3  # grid cells
DETECTION_RANGE = 2.0  # meters

def main():
    # Create output directory
    os.makedirs("results/planner_visualization", exist_ok=True)
    
    # Initialize the centralized planner
    planner = CentralizedPlanner(
        env_width=ENV_WIDTH,
        env_length=ENV_LENGTH,
        cell_size=CELL_SIZE,
        height_threshold=HEIGHT_THRESHOLD,
        fov_radius=FOV_RADIUS,
        detection_range=DETECTION_RANGE
    )
    
    # Add drones at different initial positions
    planner.add_drone(drone_id=0, initial_x=1.0, initial_y=1.0, initial_z=HEIGHT_THRESHOLD)
    planner.add_drone(drone_id=1, initial_x=8.0, initial_y=1.0, initial_z=HEIGHT_THRESHOLD)
    planner.add_drone(drone_id=2, initial_x=5.0, initial_y=8.0, initial_z=HEIGHT_THRESHOLD)
    
    # Add objects of interest
    planner.add_object_of_interest(object_id=0, x=3.0, y=3.0, z=0.0, object_type=ObjectType.PERSON)
    planner.add_object_of_interest(object_id=1, x=7.0, y=7.0, z=0.0, object_type=ObjectType.FIRE)
    planner.add_object_of_interest(object_id=2, x=2.0, y=7.0, z=0.0, object_type=ObjectType.VEHICLE)
    
    # Add obstacles
    planner.add_obstacle(object_id=0, x=5.0, y=5.0, z=0.5, width=1.0, length=1.0, height=1.0)
    planner.add_obstacle(object_id=1, x=8.0, y=3.0, z=0.5, width=1.0, length=2.0, height=1.0)
    
    # Simulate drone movement and object detection
    print("Simulating drone movement and object detection...")
    
    # Initial visualization
    planner.visualize_state()
    
    # Simulate 10 steps of movement
    for step in range(10):
        print(f"Step {step+1}/10")
        
        # Move drones (simulated movement)
        if step < 5:
            # Drone 0 moves towards (3, 3)
            planner.update_drone_position(0, 1.0 + step * 0.4, 1.0 + step * 0.4, HEIGHT_THRESHOLD)
            
            # Drone 1 moves towards (7, 7)
            planner.update_drone_position(1, 8.0 - step * 0.2, 1.0 + step * 1.2, HEIGHT_THRESHOLD)
            
            # Drone 2 stays in place
            planner.update_drone_position(2, 5.0, 8.0, HEIGHT_THRESHOLD)
        else:
            # Drone 0 continues towards (3, 3)
            planner.update_drone_position(0, 1.0 + step * 0.4, 1.0 + step * 0.4, HEIGHT_THRESHOLD)
            
            # Drone 1 continues towards (7, 7)
            planner.update_drone_position(1, 8.0 - step * 0.2, 1.0 + step * 1.2, HEIGHT_THRESHOLD)
            
            # Drone 2 moves towards (2, 7)
            planner.update_drone_position(2, 5.0 - (step - 5) * 0.6, 8.0 - (step - 5) * 0.2, HEIGHT_THRESHOLD)
        
        # Process observations (update the world map)
        new_objects_discovered = planner.process_observations()
        if new_objects_discovered:
            print(f"  New objects discovered: {planner.newly_discovered_oois}")
        
        # Visualize the current state
        planner.visualize_state()
        
        # Small delay to make the visualization more readable
        time.sleep(0.5)
    
    print("Simulation complete. Visualizations saved to results/planner_visualization/")
    
    # Show the final state
    plt.figure(figsize=(10, 10))
    planner.visualize_state(save=False)
    plt.show()

if __name__ == "__main__":
    main() 