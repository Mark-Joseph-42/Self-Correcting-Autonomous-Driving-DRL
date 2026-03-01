import numpy as np
import cv2

def discretize_semantic(raw_image):
    """
    Converts CARLA semantic segmentation raw image to a normalized grid.
    Labels for CARLA 0.9.13:
    None: 0, Roads: 1, Sidewalks: 2, Buildings: 3, Walls: 4, Fences: 5, Poles: 6, 
    TrafficLight: 7, TrafficSign: 8, Vegetation: 9, Terrain: 10, Sky: 11, 
    Pedestrians: 11, Riders: 12, Vehicles: 10, Walls: 4, ...
    Actually, let's use a simpler mapping for the agent:
    1.0: Road
    0.8: Vehicle
    0.5: Pedestrian
    0.0: Other
    """
    # CARLA semantic camera stores tags in the Red channel
    tags = raw_image[:, :, 0]
    
    grid = np.zeros_like(tags, dtype=np.float32)
    
    # Road: 1
    grid[tags == 1] = 1.0
    # Roadlines: 24 (if available) or just stick to road
    grid[tags == 24] = 1.0
    # Vehicles: 10
    grid[tags == 10] = 0.8
    # Pedestrians: 4
    grid[tags == 4] = 0.5
    
    return grid

def apply_radius_mask(grid, radius_px):
    """Masks out everything outside a given radius from the center."""
    h, w = grid.shape
    center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius_px
    return grid * mask

def compute_road_ratio(grid):
    """Simple check for forward-facing road presence."""
    h, w = grid.shape
    # Look at a wedge in front of the car
    # For a 64x64 BEV centered on car, car is at (32, 32) looking 'up'
    front_wedge = grid[0:32, 28:36]
    return np.mean(front_wedge >= 1.0)
