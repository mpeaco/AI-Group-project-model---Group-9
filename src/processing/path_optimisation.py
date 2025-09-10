import math
from typing import List, Dict


# class for a point in the cutting path
class CuttingPoint:
    def __init__(self, x, y, is_pierce=False, operation_type="cut"):
        # Basic properties of a point
        self.x = x  # x coordinate
        self.y = y  # y coordinate
        self.is_pierce = is_pierce  
        self.operation_type = operation_type  
    
    # Calculate distance between two points using pythagoras
    def distance_to(self, other):
        # Old way I found
        # dx = self.x - other.x
        # dy = self.y - other.y
        # dist = math.sqrt(dx*dx + dy*dy)
        
        # direct way:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


# Class to represent a cutting path
class CuttingPath:
    def __init__(self, points, index, is_closed=False, priority=1):
        # List of points in the path
        self.points = points
        # Is the path closed (first point connects to last point)
        self.is_closed = is_closed
        # Priority of cutting (1 is highest)
        self.priority = priority
        self.index = index
    
    def getIndex(self):
        return self.index
    
    # Calculate the total length of the path
    def length(self):
        """Calculate total path length"""
        # If don't have at least 2 points, return 0
        if len(self.points) < 2:
            return 0.0
        
        # Add up all the distances between consecutive points
        total = 0.0
        for i in range(len(self.points) - 1):
            total += self.points[i].distance_to(self.points[i + 1])
        
        # If the path is closed, add the distance from last point back to first
        if self.is_closed and len(self.points) > 2:
            total += self.points[-1].distance_to(self.points[0])
        
        return total


class PathOptimizer:
    # Optimizer for cutting paths 

    def __init__(self):
        # PathOptimizer now focuses only on path optimization
        # Time calculations are handled by CuttingDepthManager with material profiles
        pass
    
    # Find the best order of paths
    def nearest_neighbor_tsp(self, paths):
        # Nothing to do for 0 or 1 paths
        if len(paths) <= 1:
            return paths
        
        result = []
        paths_left = paths.copy()
        current_pos = CuttingPoint(0, 0)  
        
        # Process all paths
        while paths_left:
            # Find nearest path
            nearest_idx = 0
            min_dist = float('inf')
            
            # Check each path
            for i in range(len(paths_left)):
                path = paths_left[i]
                if not path.points:
                    continue

                # Distance calculation
                print("POINTS", path.points)
                start = path.points[0]
                dist = math.sqrt((current_pos.x - start.x)**2 + (current_pos.y - start.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            # Store nearest path to result
            best_path = paths_left[nearest_idx]
            result.append(best_path)
            
            # Update position 
            if best_path.points:
                current_pos = best_path.points[-1]
            paths_left.pop(nearest_idx)
        
        return result
        
    def add_lead_in_out(self, path, lead_length=5.0, lead_angle=45.0):
        """
        Add simple lead-in and lead-out segments to a cutting path.
        
        Args:
            path (CuttingPath): The original cutting path
            lead_length (float): Length of the lead-in/out segment in mm
            lead_angle (float): Angle of approach/exit in degrees
            
        Returns:
            CuttingPath: Path with added lead-in/out points
        """
        # If path is empty or too short, return unchanged
        if not path.points or len(path.points) < 2:
            return path
            
        # Create new points list
        new_points = []
        
        # Get first point
        first_point = path.points[0]
        
        # Only add lead-in/out for closed paths
        if path.is_closed:
            # Simple approach - add points at 45 degrees from first point
            # Calculate lead-in point 
            lead_in_x = first_point.x - lead_length
            lead_in_y = first_point.y - lead_length
            lead_in = CuttingPoint(lead_in_x, lead_in_y, is_pierce=True)
            
            # Calculate lead-out point (5mm away at opposite 45 degrees)
            lead_out_x = first_point.x + lead_length
            lead_out_y = first_point.y - lead_length
            lead_out = CuttingPoint(lead_out_x, lead_out_y)
            
            # Build new path: lead-in -> original points -> lead-out
            new_points.append(lead_in)
            new_points.extend(path.points)
            new_points.append(lead_out)
        else:
            # For open paths, just mark first point as pierce point
            first_point.is_pierce = True
            new_points = path.points
        
        # Return new path with same properties
        return CuttingPath(new_points, path.is_closed, path.priority)


def process_path(path):
    # Convert SVG path to cutting path
    points = []
    
    # Extract points from segments
    for segment in path:
        # Create points for start and end
        start = CuttingPoint(segment.start.real, segment.start.imag)
        end = CuttingPoint(segment.end.real, segment.end.imag)
        
        # Add points to the list
        points.append(start)
        points.append(end)
    
    # Check if path is closed
    is_closed = False
    if len(points) > 2:
        # If first and last points are close then is_closed = True
        first = points[0]
        last = points[-1]
        dist = math.sqrt((first.x - last.x)**2 + (first.y - last.y)**2)
        if dist < 1.0:
            is_closed = True
    
    # Create cutting path object
    return CuttingPath(points, is_closed)


def optimize_cutting_sequence(paths, method="nearest_neighbor"):
    # Optimize the order of cutting paths
    optimizer = PathOptimizer()
    return optimizer.nearest_neighbor_tsp(paths)
