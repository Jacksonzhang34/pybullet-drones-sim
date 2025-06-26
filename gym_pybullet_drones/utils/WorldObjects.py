import numpy as np
from enum import Enum

class ObjectType(Enum):
    """Enum for the types of objects in the environment."""
    PERSON = 0
    FIRE = 1
    VEHICLE = 2
    OBSTACLE = 3


class WorldObject:
    """Base class for objects in the environment."""
    
    def __init__(self, x: float, y: float, z: float, object_id: int):
        """Initialize a world object.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            z: Z-coordinate in meters
            object_id: Unique identifier for the object
        """
        self.x = x
        self.y = y
        self.z = z
        self.object_id = object_id
        
    def get_position(self):
        """Get the position of the object.
        
        Returns:
            tuple: (x, y, z) position in meters
        """
        return self.x, self.y, self.z


class ObjectOfInterest(WorldObject):
    """Class representing an object of interest in the environment."""
    
    def __init__(self, x: float, y: float, z: float, object_id: int, object_type: ObjectType):
        """Initialize an object of interest.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            z: Z-coordinate in meters
            object_id: Unique identifier for the object
            object_type: Type of the object of interest
        """
        super().__init__(x, y, z, object_id)
        self.object_type = object_type
        self.discovered = False
        self.assigned_drone = None
        
    def discover(self):
        """Mark the object as discovered."""
        self.discovered = True
        
    def assign_drone(self, drone_id: int):
        """Assign a drone to this object.
        
        Args:
            drone_id: ID of the drone assigned to this object
        """
        self.assigned_drone = drone_id


class Obstacle(WorldObject):
    """Class representing an obstacle in the environment."""
    
    def __init__(self, x: float, y: float, z: float, object_id: int, width: float, length: float, height: float):
        """Initialize an obstacle.
        
        Args:
            x: X-coordinate of the center in meters
            y: Y-coordinate of the center in meters
            z: Z-coordinate of the center in meters
            object_id: Unique identifier for the obstacle
            width: Width of the obstacle in meters (x-axis)
            length: Length of the obstacle in meters (y-axis)
            height: Height of the obstacle in meters (z-axis)
        """
        super().__init__(x, y, z, object_id)
        self.width = width
        self.length = length
        self.height = height
        
    def is_point_inside(self, x: float, y: float, z: float) -> bool:
        """Check if a point is inside the obstacle.
        
        Args:
            x: X-coordinate to check
            y: Y-coordinate to check
            z: Z-coordinate to check
            
        Returns:
            bool: True if the point is inside the obstacle, False otherwise
        """
        half_width = self.width / 2
        half_length = self.length / 2
        half_height = self.height / 2
        
        return (self.x - half_width <= x <= self.x + half_width and
                self.y - half_length <= y <= self.y + half_length and
                self.z - half_height <= z <= self.z + half_height) 