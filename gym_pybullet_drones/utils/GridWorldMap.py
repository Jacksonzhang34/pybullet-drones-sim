import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class CellState(Enum):
    """Enum for the possible states of a cell in the grid world."""
    UNKNOWN = 0
    EXPLORED_EMPTY = 1
    OBSTACLE = 2
    OBJECT_OF_INTEREST = 3


class GridWorldMap:
    """A 2D grid representation of the environment for high-level planning.
    
    This class maintains a discretized map of the environment where each cell
    can be in one of several states (unknown, explored and empty, obstacle,
    or containing an object of interest).
    """
    
    def __init__(self, 
                 width: float, 
                 length: float, 
                 cell_size: float,
                 height_threshold: float):
        """Initialize the grid world map.
        
        Args:
            width: Width of the environment in meters
            length: Length of the environment in meters
            cell_size: Size of each grid cell in meters
            height_threshold: The fixed height at which drones operate
        """
        self.width = width
        self.length = length
        self.cell_size = cell_size
        self.height_threshold = height_threshold
        
        # Calculate grid dimensions
        self.grid_width = int(np.ceil(width / cell_size))
        self.grid_length = int(np.ceil(length / cell_size))
        
        # Initialize grid with all cells as UNKNOWN
        self.grid = np.full((self.grid_width, self.grid_length), 
                           CellState.UNKNOWN.value, 
                           dtype=int)
        
        # Keep track of objects of interest positions
        self.ooi_positions = []  # List of (grid_x, grid_y) positions
        
        # Keep track of obstacle positions
        self.obstacle_positions = []  # List of (grid_x, grid_y) positions
        
    def continuous_to_grid(self, x: float, y: float) -> tuple:
        """Convert continuous coordinates to grid coordinates.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            
        Returns:
            tuple: (grid_x, grid_y) coordinates
        """
        grid_x = min(self.grid_width - 1, max(0, int(x / self.cell_size)))
        grid_y = min(self.grid_length - 1, max(0, int(y / self.cell_size)))
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> tuple:
        """Convert grid coordinates to continuous coordinates (cell center).
        
        Args:
            grid_x: Grid X-coordinate
            grid_y: Grid Y-coordinate
            
        Returns:
            tuple: (x, y, z) coordinates in meters
        """
        x = (grid_x + 0.5) * self.cell_size
        y = (grid_y + 0.5) * self.cell_size
        z = self.height_threshold
        return x, y, z
    
    def update_cell(self, x: float, y: float, state: CellState):
        """Update the state of a cell based on continuous coordinates.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            state: New state of the cell
        """
        grid_x, grid_y = self.continuous_to_grid(x, y)
        self.grid[grid_x, grid_y] = state.value
        
        # Keep track of special cells
        if state == CellState.OBJECT_OF_INTEREST:
            if (grid_x, grid_y) not in self.ooi_positions:
                self.ooi_positions.append((grid_x, grid_y))
        elif state == CellState.OBSTACLE:
            if (grid_x, grid_y) not in self.obstacle_positions:
                self.obstacle_positions.append((grid_x, grid_y))
    
    def get_cell_state(self, x: float, y: float) -> CellState:
        """Get the state of a cell based on continuous coordinates.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            
        Returns:
            CellState: The state of the cell
        """
        grid_x, grid_y = self.continuous_to_grid(x, y)
        return CellState(self.grid[grid_x, grid_y])
    
    def update_fov(self, drone_x: float, drone_y: float, fov_radius: int):
        """Update cells within the drone's field of view as explored.
        
        Args:
            drone_x: Drone's X-coordinate in meters
            drone_y: Drone's Y-coordinate in meters
            fov_radius: Field of view radius in grid cells
        """
        grid_x, grid_y = self.continuous_to_grid(drone_x, drone_y)
        
        # Update cells within the FOV radius
        for dx in range(-fov_radius, fov_radius + 1):
            for dy in range(-fov_radius, fov_radius + 1):
                # Check if the cell is within the circular FOV
                if dx**2 + dy**2 <= fov_radius**2:
                    nx, ny = grid_x + dx, grid_y + dy
                    
                    # Check if the cell is within grid boundaries
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_length:
                        # Only update if the cell is UNKNOWN
                        if self.grid[nx, ny] == CellState.UNKNOWN.value:
                            self.grid[nx, ny] = CellState.EXPLORED_EMPTY.value
    
    def is_within_bounds(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are within the map bounds.
        
        Args:
            grid_x: Grid X-coordinate
            grid_y: Grid Y-coordinate
            
        Returns:
            bool: True if coordinates are within bounds, False otherwise
        """
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_length
    
    def visualize(self, drone_positions=None):
        """Visualize the grid world map.
        
        Args:
            drone_positions: List of (x, y, z) drone positions in meters
        """
        plt.figure(figsize=(10, 10))
        
        # Create a colormap for the grid
        cmap = plt.cm.colors.ListedColormap(['lightgray', 'white', 'black', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the grid
        plt.imshow(self.grid.T, cmap=cmap, norm=norm, origin='lower')
        
        # Plot drone positions if provided
        if drone_positions:
            for i, pos in enumerate(drone_positions):
                x, y = self.continuous_to_grid(pos[0], pos[1])
                plt.plot(x, y, 'bo', markersize=10, label=f'Drone {i+1}' if i == 0 else "")
        
        # Add colorbar
        cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['Unknown', 'Empty', 'Obstacle', 'Object of Interest'])
        
        plt.title('Grid World Map')
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        if drone_positions:
            plt.legend()
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure for saving or displaying 