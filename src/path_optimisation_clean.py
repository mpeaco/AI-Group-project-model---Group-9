import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class CuttingPoint:
    """Represents a point in the cutting path with metadata"""
    x: float
    y: float
    is_pierce: bool = False
    operation_type: str = "cut"  # cut, engrave, mark
    
    def distance_to(self, other: 'CuttingPoint') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class CuttingPath:
    """Represents a complete cutting path with optimization data"""
    points: List[CuttingPoint]
    is_closed: bool = False
    priority: int = 1  # 1 = highest priority
    
    def length(self) -> float:
        """Calculate total path length"""
        if len(self.points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.points) - 1):
            total += self.points[i].distance_to(self.points[i + 1])
        
        if self.is_closed and len(self.points) > 2:
            total += self.points[-1].distance_to(self.points[0])
        
        return total


class PathOptimizer:
    """Main class for optimizing cutting paths for laser cutting machines"""
    
    def __init__(self, machine_config: Dict = None):
        self.machine_config = machine_config or {
            'max_speed': 1000,  # mm/min
            'pierce_time': 0.5,  # seconds
            'travel_speed': 5000,  # mm/min
            'lead_in_length': 2.0,  # mm
        }
    
    def nearest_neighbor_tsp(self, paths: List[CuttingPath]) -> List[CuttingPath]:
        """
        Solve TSP using nearest neighbor algorithm for path ordering
        
        Args:
            paths: List of cutting paths to optimize
            
        Returns:
            Optimized list of paths
        """
        if len(paths) <= 1:
            return paths
        
        optimized_paths = []
        remaining_paths = paths.copy()
        current_position = CuttingPoint(0, 0)  # Assume origin start
        
        while remaining_paths:
            # Find nearest path
            min_distance = float('inf')
            nearest_path = None
            nearest_idx = -1
            
            for idx, path in enumerate(remaining_paths):
                if not path.points:
                    continue
                
                # Check distance to start and end of path
                start_dist = current_position.distance_to(path.points[0])
                end_dist = current_position.distance_to(path.points[-1])
                
                # Choose closest point
                if start_dist < min_distance:
                    min_distance = start_dist
                    nearest_path = path
                    nearest_idx = idx
                    reverse_path = False
                
                if end_dist < min_distance:
                    min_distance = end_dist
                    nearest_path = path
                    nearest_idx = idx
                    reverse_path = True
            
            if nearest_path:
                # Reverse path if needed for optimal direction
                if reverse_path:
                    nearest_path.points.reverse()
                
                optimized_paths.append(nearest_path)
                current_position = nearest_path.points[-1]
                remaining_paths.pop(nearest_idx)
        
        return optimized_paths
    
    def add_lead_in_out(self, path: CuttingPath) -> CuttingPath:
        """
        Add lead-in and lead-out paths for clean cutting
        
        Args:
            path: Original cutting path
            
        Returns:
            Path with lead-in/out added
        """
        if len(path.points) < 2:
            return path
        
        lead_length = self.machine_config['lead_in_length']
        
        # Calculate lead-in direction (perpendicular to first segment)
        first_point = path.points[0]
        second_point = path.points[1]
        
        dx = second_point.x - first_point.x
        dy = second_point.y - first_point.y
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Perpendicular vector for lead-in
            perp_x = -dy / length * lead_length
            perp_y = dx / length * lead_length
            
            lead_in_point = CuttingPoint(
                first_point.x + perp_x,
                first_point.y + perp_y,
                is_pierce=True
            )
            
            # Add lead-in
            path.points.insert(0, lead_in_point)
            
            # Add lead-out (same but at end)
            last_point = path.points[-1]
            second_last = path.points[-2]
            
            dx = last_point.x - second_last.x
            dy = last_point.y - second_last.y
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                perp_x = dy / length * lead_length
                perp_y = -dx / length * lead_length
                
                lead_out_point = CuttingPoint(
                    last_point.x + perp_x,
                    last_point.y + perp_y
                )
                
                path.points.append(lead_out_point)
        
        return path
    
    def calculate_cutting_time(self, paths: List[CuttingPath]) -> Dict:
        """
        Calculate estimated cutting time for all paths
        
        Args:
            paths: List of optimized cutting paths
            
        Returns:
            Dictionary with time breakdown
        """
        total_cut_time = 0.0
        total_travel_time = 0.0
        total_pierce_time = 0.0
        pierce_count = 0
        
        current_pos = CuttingPoint(0, 0)
        
        for path in paths:
            if not path.points:
                continue
            
            # Travel time to start of path
            travel_distance = current_pos.distance_to(path.points[0])
            total_travel_time += travel_distance / self.machine_config['travel_speed'] * 60
            
            # Pierce time
            total_pierce_time += self.machine_config['pierce_time']
            pierce_count += 1
            
            # Cutting time
            cut_distance = path.length()
            total_cut_time += cut_distance / self.machine_config['max_speed'] * 60
            
            current_pos = path.points[-1]
        
        return {
            'total_time': total_cut_time + total_travel_time + total_pierce_time,
            'cutting_time': total_cut_time,
            'travel_time': total_travel_time,
            'pierce_time': total_pierce_time,
            'pierce_count': pierce_count,
            'total_distance': sum(path.length() for path in paths)
        }


def visualize_paths(original_paths: List[CuttingPath], 
                   optimized_paths: List[CuttingPath] = None, 
                   processed_image = None,
                   save_path: str = "path_visualization.png"):
    """
    Visualize original and optimized paths side by side
    """
    n_plots = 2 if optimized_paths else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot original paths
    ax1 = axes[0]
    if processed_image is not None:
        ax1.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
    
    for i, path in enumerate(original_paths):
        if not path.points:
            continue
            
        x_coords = [p.x for p in path.points]
        y_coords = [p.y for p in path.points]
        
        color = colors[i % len(colors)]
        ax1.plot(x_coords, y_coords, color=color, linewidth=2, label=f'Path {i+1}')
        ax1.scatter(path.points[0].x, path.points[0].y, 
                   color=color, s=80, marker='o', edgecolor='black')
    
    ax1.set_title('Original Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
