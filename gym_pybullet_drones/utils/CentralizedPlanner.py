import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque
import os

from gym_pybullet_drones.utils.GridWorldMap import GridWorldMap, CellState
from gym_pybullet_drones.utils.WorldObjects import WorldObject, ObjectOfInterest, Obstacle, ObjectType

class DroneState:
    """Class representing the state of a drone in the high-level planner."""
    
    def __init__(self, drone_id: int, initial_x: float, initial_y: float, initial_z: float):
        """Initialize a drone state.
        
        Args:
            drone_id: Unique identifier for the drone
            initial_x: Initial X-coordinate in meters
            initial_y: Initial Y-coordinate in meters
            initial_z: Initial Z-coordinate in meters
        """
        self.drone_id = drone_id
        self.x = initial_x
        self.y = initial_y
        self.z = initial_z
        self.assigned_ooi = None  # ID of the assigned object of interest
        self.current_path = deque([])  #Deque of (grid_x, grid_y) waypoints
        
    def update_position(self, x: float, y: float, z: float):
        """Update the drone's position.
        
        Args:
            x: New X-coordinate in meters
            y: New Y-coordinate in meters
            z: New Z-coordinate in meters
        """
        self.x = x
        self.y = y
        self.z = z
        
    def assign_ooi(self, ooi_id: int):
        """Assign an object of interest to this drone.
        
        Args:
            ooi_id: ID of the object of interest
        """
        self.assigned_ooi = ooi_id
        
    def set_path(self, path: Deque[Tuple[int, int]]):
        """Set the path for the drone to follow.
        
        Args:
            path: List of (grid_x, grid_y) waypoints
        """
        self.current_path = path
        
    def get_next_waypoint(self) -> Optional[Tuple[int, int]]:
        """Get the next waypoint in the path.
        
        Returns:
            Optional[Tuple[int, int]]: The next waypoint, or None if the path is empty
        """
        if self.current_path:
            return self.current_path[0]
        return None
    
    def advance_path(self):
        """Remove the first waypoint from the path after reaching it."""
        if self.current_path:
            self.current_path.popleft()


class CentralizedPlanner:
    """Centralized planner for coordinating multiple drones."""
    
    def __init__(self, 
                 env_width: float, 
                 env_length: float, 
                 cell_size: float,
                 height_threshold: float,
                 fov_radius: int,
                 detection_range: float):
        """Initialize the centralized planner.
        
        Args:
            env_width: Width of the environment in meters
            env_length: Length of the environment in meters
            cell_size: Size of each grid cell in meters
            height_threshold: The fixed height at which drones operate
            fov_radius: Field of view radius in grid cells
            detection_range: Range within which drones can detect objects in meters
        """
        # Initialize the grid world map
        self.grid_map = GridWorldMap(env_width, env_length, cell_size, height_threshold)
        
        # fov_radius should be the same as detection_range
        self.fov_radius = fov_radius 
        self.detection_range = detection_range
        
        # Drones, objects, and obstacles
        self.drones: Dict[int, DroneState] = {}
        self.oois: Dict[int, ObjectOfInterest] = {}
        self.obstacles: Dict[int, Obstacle] = {}
        
        # Tracking discovered objects
        self.newly_discovered_oois: List[int] = []
        
        # For visualization
        self.visualization_dir = "results/planner_visualization"
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.step_counter = 0
        
    def add_drone(self, drone_id: int, initial_x: float, initial_y: float, initial_z: float):
        """Add a drone to the planner.
        
        Args:
            drone_id: Unique identifier for the drone
            initial_x: Initial X-coordinate in meters
            initial_y: Initial Y-coordinate in meters
            initial_z: Initial Z-coordinate in meters
        """
        if drone_id in self.drones:
            raise ValueError(f"Drone with id {drone_id} already exists")
        self.drones[drone_id] = DroneState(drone_id, initial_x, initial_y, initial_z)
        
        # Mark the initial position as explored in the grid
        grid_x, grid_y = self.grid_map.continuous_to_grid(initial_x, initial_y)
        self.grid_map.grid[grid_x, grid_y] = CellState.EXPLORED_EMPTY.value
        
    def add_object_of_interest(self, 
                              object_id: int, 
                              x: float, 
                              y: float, 
                              z: float, 
                              object_type: ObjectType):
        """Add an object of interest to the environment.
        
        Args:
            object_id: Unique identifier for the object
            x: X-coordinate in meters
            y: Y-coordinate in meters
            z: Z-coordinate in meters
            object_type: Type of the object
        """
        if object_id in self.oois:
            raise ValueError(f"OOI with id {object_id} already exists")
        self.oois[object_id] = ObjectOfInterest(x, y, z, object_id, object_type)
        
    def add_obstacle(self, 
                    object_id: int, 
                    x: float, 
                    y: float, 
                    z: float, 
                    width: float, 
                    length: float, 
                    height: float):
        """Add an obstacle to the environment.
        
        Args:
            object_id: Unique identifier for the obstacle
            x: X-coordinate of the center in meters
            y: Y-coordinate of the center in meters
            z: Z-coordinate of the center in meters
            width: Width of the obstacle in meters (x-axis)
            length: Length of the obstacle in meters (y-axis)
            height: Height of the obstacle in meters (z-axis)
        """
        if object_id in self.obstacles:
            raise ValueError(f"Obstacle with id {object_id} already exists")

        self.obstacles[object_id] = Obstacle(x, y, z, object_id, width, length, height)
        
        # Mark the obstacle in the grid
        grid_x, grid_y = self.grid_map.continuous_to_grid(x, y)
        self.grid_map.update_cell(x, y, CellState.OBSTACLE)
        
    def update_drone_position(self, drone_id: int, x: float, y: float, z: float):
        """Update a drone's position.
        
        Args:
            drone_id: ID of the drone
            x: New X-coordinate in meters
            y: New Y-coordinate in meters
            z: New Z-coordinate in meters
        """
        if drone_id in self.drones:
            self.drones[drone_id].update_position(x, y, z)
            
    def process_observations(self):
        """Process observations from all drones to update the world map.
        
        This function simulates the detection of objects and obstacles by drones
        based on their current positions and field of view.
        """
        self.newly_discovered_oois = []
        
        for drone_id, drone in self.drones.items():
            # Update the grid with cells in the drone's FOV
            self.grid_map.update_fov(drone.x, drone.y, self.fov_radius)
            
            # Check for objects of interest within detection range
            for ooi_id, ooi in self.oois.items():
                if not ooi.discovered:
                    # Calculate distance to the object
                    distance = np.sqrt((drone.x - ooi.x)**2 + (drone.y - ooi.y)**2)
                    
                    if distance <= self.detection_range:
                        # Object is detected
                        ooi.discover()
                        self.newly_discovered_oois.append(ooi_id)
                        
                        # Update the grid map
                        self.grid_map.update_cell(ooi.x, ooi.y, CellState.OBJECT_OF_INTEREST)
            
            # Check for obstacles within detection range
            for obs_id, obs in self.obstacles.items():
                # Calculate distance to the obstacle center
                distance = np.sqrt((drone.x - obs.x)**2 + (drone.y - obs.y)**2)
                
                if distance <= self.detection_range + max(obs.width, obs.length)/2:
                    # Obstacle is detected, update the grid map
                    self.grid_map.update_cell(obs.x, obs.y, CellState.OBSTACLE)
        
        return len(self.newly_discovered_oois) > 0
    
    def visualize_state(self, save=True):
        """Visualize the current state of the environment.
        
        Args:
            save: Whether to save the visualization to a file
        
        Returns:
            plt.Figure: The matplotlib figure object
        """
        # Get drone positions for visualization
        drone_positions = [(drone.x, drone.y, drone.z) for drone in self.drones.values()]
        
        # Create the visualization
        fig = self.grid_map.visualize(drone_positions)
        
        # Mark objects of interest
        for ooi in self.oois.values():
            if ooi.discovered:
                grid_x, grid_y = self.grid_map.continuous_to_grid(ooi.x, ooi.y)
                plt.plot(grid_x, grid_y, 'r*', markersize=10)
                
                # If assigned to a drone, draw a line
                if ooi.assigned_drone is not None:
                    drone = self.drones[ooi.assigned_drone]
                    drone_x, drone_y = self.grid_map.continuous_to_grid(drone.x, drone.y)
                    plt.plot([drone_x, grid_x], [drone_y, grid_y], 'r--', alpha=0.7)
        
        # Save the figure if requested
        if save:
            plt.savefig(f"{self.visualization_dir}/step_{self.step_counter:04d}.png")
            self.step_counter += 1
        
        return fig 