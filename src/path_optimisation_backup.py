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
    estimated_time: float = 0.0
    
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


def visualize_paths(original_paths: List[CuttingPath], 
                   optimized_paths: List[CuttingPath] = None, 
                   processed_image: np.ndarray = None,
                   save_path: str = "path_visualization.png"):
    """
    Visualize the cutting paths and optimization results
    
    Args:
        original_paths: Original cutting paths
        optimized_paths: Optimized cutting paths (optional)
        processed_image: Processed binary image (optional)
        save_path: Path to save the visualization
    """
    fig_width = 15 if optimized_paths else 10
    fig, axes = plt.subplots(1, 2 if optimized_paths else 1, figsize=(fig_width, 8))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot original paths
    ax1 = axes[0]
    
    # Show processed image as background if provided
    if processed_image is not None:
        ax1.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
    
    # Plot original paths
    for i, path in enumerate(original_paths):
        if not path.points:
            continue
            
        x_coords = [p.x for p in path.points]
        y_coords = [p.y for p in path.points]
        
        color = colors[i % len(colors)]
        ax1.plot(x_coords, y_coords, color=color, linewidth=2, 
                label=f'Path {i+1} ({len(path.points)} pts)')
        
        # Mark start and end points
        if path.points:
            ax1.scatter(path.points[0].x, path.points[0].y, 
                       color=color, s=100, marker='o', edgecolor='black', linewidth=2)
            ax1.scatter(path.points[-1].x, path.points[-1].y, 
                       color=color, s=100, marker='s', edgecolor='black', linewidth=2)
    
    ax1.set_title('Original Paths', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot optimized paths if provided
    if optimized_paths:
        ax2 = axes[1]
        
        # Show processed image as background
        if processed_image is not None:
            ax2.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
        
        # Plot optimized paths with travel moves
        current_pos = CuttingPoint(0, 0)  # Start at origin
        
        for i, path in enumerate(optimized_paths):
            if not path.points:
                continue
                
            x_coords = [p.x for p in path.points]
            y_coords = [p.y for p in path.points]
            
            color = colors[i % len(colors)]
            
            # Draw travel move (dashed line)
            if i > 0 or (current_pos.x != path.points[0].x or current_pos.y != path.points[0].y):
                ax2.plot([current_pos.x, path.points[0].x], 
                        [current_pos.y, path.points[0].y], 
                        'k--', alpha=0.5, linewidth=1, label='Travel' if i == 0 else "")
            
            # Draw cutting path
            ax2.plot(x_coords, y_coords, color=color, linewidth=2, 
                    label=f'Path {i+1}')
            
            # Mark start and end points
            ax2.scatter(path.points[0].x, path.points[0].y, 
                       color=color, s=100, marker='o', edgecolor='black', linewidth=2)
            ax2.scatter(path.points[-1].x, path.points[-1].y, 
                       color=color, s=100, marker='s', edgecolor='black', linewidth=2)
            
            # Add path number
            center_x = sum(p.x for p in path.points) / len(path.points)
            center_y = sum(p.y for p in path.points) / len(path.points)
            ax2.text(center_x, center_y, str(i+1), 
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            current_pos = path.points[-1]
        
        ax2.set_title('Optimized Paths (with cutting sequence)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Path visualization saved to: {save_path}")
    plt.show()


def create_cutting_sequence_animation(optimized_paths: List[CuttingPath], 
                                    processed_image: np.ndarray = None,
                                    save_path: str = "cutting_sequence.png"):
    """
    Create a step-by-step visualization of the cutting sequence
    
    Args:
        optimized_paths: Optimized cutting paths
        processed_image: Processed binary image (optional)
        save_path: Path to save the animation frames
    """
    n_paths = len(optimized_paths)
    cols = min(4, n_paths)
    rows = (n_paths + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    current_pos = CuttingPoint(0, 0)
    
    for step in range(n_paths):
        ax = axes[step]
        
        # Show processed image as background
        if processed_image is not None:
            ax.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
        
        # Draw all previous paths in gray
        for i in range(step):
            path = optimized_paths[i]
            x_coords = [p.x for p in path.points]
            y_coords = [p.y for p in path.points]
            ax.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.5)
        
        # Draw current path in color
        current_path = optimized_paths[step]
        x_coords = [p.x for p in current_path.points]
        y_coords = [p.y for p in current_path.points]
        
        color = colors[step % len(colors)]
        ax.plot(x_coords, y_coords, color=color, linewidth=3)
        
        # Draw travel move to this path
        if step > 0:
            prev_end = optimized_paths[step-1].points[-1]
            ax.plot([prev_end.x, current_path.points[0].x], 
                   [prev_end.y, current_path.points[0].y], 
                   'k--', alpha=0.7, linewidth=2)
        
        # Mark start point
        ax.scatter(current_path.points[0].x, current_path.points[0].y, 
                  color='green', s=150, marker='o', edgecolor='black', linewidth=2)
        
        ax.set_title(f'Step {step+1}: Path {step+1}\nLength: {current_path.length():.1f}mm', 
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_paths, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cutting sequence visualization saved to: {save_path}")
    plt.show()


class PathOptimizer:
    """Main class for optimizing cutting paths for laser cutting machines"""
    
    def __init__(self, machine_config: Dict = None):
        self.machine_config = machine_config or {
            'max_speed': 1000,  # mm/min
            'pierce_time': 0.5,  # seconds
            'travel_speed': 5000,  # mm/min
            'kerf_width': 0.1,  # mm
            'lead_in_length': 2.0,  # mm
            'lead_out_length': 2.0  # mm
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


def process_path(path):
    """
    Takes the path that has been extracted
    Processes to make it useable for optimising

    Args:
        path (svgpathtools.path.Path): path extracted from image

    Returns:
        CuttingPath: the path that has been processed for laser cutting
    """
    points = []
    
    # Convert SVG path segments to cutting points
    for segment in path:
        start_point = CuttingPoint(
            segment.start.real,
            segment.start.imag
        )
        end_point = CuttingPoint(
            segment.end.real,
            segment.end.imag
        )
        
        if not points or points[-1].distance_to(start_point) > 0.1:
            points.append(start_point)
        points.append(end_point)
    
    # Determine if path is closed
    is_closed = False
    if len(points) > 2:
        is_closed = points[0].distance_to(points[-1]) < 1.0
    
    cutting_path = CuttingPath(points=points, is_closed=is_closed)
    
    return cutting_path


def optimize_cutting_sequence(paths: List[CuttingPath], 
                             method: str = "nearest_neighbor") -> List[CuttingPath]:
    """
    Optimize the sequence of cutting paths for minimum travel time
    
    Args:
        paths: List of cutting paths
        method: Optimization method ("nearest_neighbor")
        
    Returns:
        Optimized list of cutting paths
    """
    optimizer = PathOptimizer()
    return optimizer.nearest_neighbor_tsp(paths)


def generate_cutting_report(paths: List[CuttingPath]) -> Dict:
    """
    Generate comprehensive cutting report with metrics
    
    Args:
        paths: List of optimized cutting paths
        
    Returns:
        Dictionary with cutting analysis
    """
    optimizer = PathOptimizer()
    time_analysis = optimizer.calculate_cutting_time(paths)
    
    total_length = sum(path.length() for path in paths)
    path_count = len(paths)
    
    return {
        'path_count': path_count,
        'total_cutting_length': total_length,
        'estimated_time_minutes': time_analysis['total_time'],
        'pierce_count': time_analysis['pierce_count'],
        'paths': paths
    }

