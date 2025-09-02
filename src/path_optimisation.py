import matplotlib.pyplot as plt
import numpy as np
import math
import os
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
    def __init__(self, points, is_closed=False, priority=1):
        # List of points in the path
        self.points = points
        # Is the path closed (first point connects to last point)
        self.is_closed = is_closed
        # Priority of cutting (1 is highest)
        self.priority = priority
    
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


# Basic visualization
def visualize_paths(original_paths, optimized_paths=None, processed_image=None, save_path="path_visualization.png"):
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
    plt.close()


def create_cutting_sequence_animation(optimized_paths, processed_image=None, output_dir=None, save_path=None):
    """
    Create a step-by-step visualization of the cutting sequence

    Args:
        optimized_paths: Optimized cutting paths
        processed_image: Processed binary image (optional)
        output_dir: Directory to save the visualization
        save_path: Complete path to save the visualization (overrides output_dir)
    """
    # Determine output path
    if save_path:
        output_path = save_path
    elif output_dir:
        output_path = os.path.join(output_dir, "cutting_sequence_steps.png")
    else:
        output_path = "cutting_sequence_steps.png"
    
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cutting sequence visualization saved to: {output_path}")
    plt.close()


class PathOptimizer:
    # Optimizer for cutting paths

    def __init__(self, machine_config=None):
        # Basic settings
        if machine_config is None:
            self.machine_config = {
                'max_speed': 12000,  # mm/min
                'pierce_time': 0.5,  # seconds
                'travel_speed': 20000,  # mm/min
            }
            print("DEBUG - PathOptimizer initialized with default settings:", self.machine_config)
        else:
            self.machine_config = machine_config
            print("DEBUG - PathOptimizer initialized with custom settings:", self.machine_config)
    
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
    
    # Calculate time 
    def calculate_cutting_time(self, paths):
        # Time calculation
        total_distance = 0
        pierce_count = 0
        
        # Add up distances
        for path in paths:
            if path.points:
                total_distance += path.length()
                pierce_count += 1

        # Time calculation
        cut_time = total_distance / self.machine_config['max_speed'] * 60
        pierce_time = pierce_count * self.machine_config['pierce_time']
        total_time = cut_time + pierce_time
        
        return {
            'total_time': total_time,
            'pierce_count': pierce_count,
            'total_distance': total_distance
        }
        
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
            # Calculate lead-in point (5mm away at 45 degrees)
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


def generate_cutting_report(paths):
    # Report withmetrics
    optimizer = PathOptimizer()
    time_info = optimizer.calculate_cutting_time(paths)

    # Report dictionary
    total_length = 0
    for p in paths:
        if p.points:
            total_length += p.length()
    
    report = {
        'path_count': len(paths),
        'total_cutting_length': total_length,
        'estimated_time_minutes': time_info['total_time'],
        'pierce_count': time_info['pierce_count'],
        'paths': paths
    }
    
    return report

