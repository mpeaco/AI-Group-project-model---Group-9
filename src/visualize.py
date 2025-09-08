import matplotlib.pyplot as plt
import numpy as np
import os
from processing.path_optimisation import CuttingPoint


def visualize_paths(original_paths, optimized_paths=None, processed_image=None, save_path="path_visualization.png"):
    # Basic colors for different paths
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create figure
    if optimized_paths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))

    # Show background image
    if processed_image is not None:
        ax1.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
    
    # Draw original paths
    for i, path in enumerate(original_paths):
        if not path.points:
            continue
            
        # Extract x and y coordinates
        x_vals = [p.x for p in path.points]
        y_vals = [p.y for p in path.points]
        
        color = colors[i % len(colors)]
        
        # Draw path
        ax1.plot(x_vals, y_vals, color=color, linewidth=2, label=f'Path {i+1}')
        
        # Mark start and end
        ax1.scatter(path.points[0].x, path.points[0].y, color=color, s=100, marker='o', edgecolor='black', linewidth=2)
        ax1.scatter(path.points[-1].x, path.points[-1].y, color=color, s=100, marker='s', edgecolor='black', linewidth=2)

    # Set up first plot
    ax1.set_title('Original Paths')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Handle optimized paths
    if optimized_paths:
        # Show background
        if processed_image is not None:
            ax2.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
        
        current_x, current_y = 0, 0
        
        # Draw each optimized path
        for i, path in enumerate(optimized_paths):
            if not path.points:
                continue
            
            # Extract coordinates
            x_vals = [p.x for p in path.points]
            y_vals = [p.y for p in path.points]
            
            color = colors[i % len(colors)]
            
            # Draw travel move if needed
            start_x, start_y = path.points[0].x, path.points[0].y
            if current_x != start_x or current_y != start_y:
                travel_label = 'Travel' if i == 0 else ""
                ax2.plot([current_x, start_x], [current_y, start_y], 'k--', alpha=0.5, linewidth=1, label=travel_label)
            
            # Draw path and markers
            ax2.plot(x_vals, y_vals, color=color, linewidth=2, label=f'Path {i+1}')
            ax2.scatter(path.points[0].x, path.points[0].y, color=color, s=100, marker='o', edgecolor='black', linewidth=2)
            ax2.scatter(path.points[-1].x, path.points[-1].y, color=color, s=100, marker='s', edgecolor='black', linewidth=2)
            
            # Add path number in the middle
            center_x = sum(p.x for p in path.points) / len(path.points)
            center_y = sum(p.y for p in path.points) / len(path.points)
            
            ax2.text(center_x, center_y, str(i+1), fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Update current position
            current_x, current_y = path.points[-1].x, path.points[-1].y

        # Set up second plot
        ax2.set_title('Optimized Paths')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Saved:", save_path)
    plt.close()


def create_cutting_sequence_animation(optimized_paths, processed_image=None, output_dir=None, save_path=None):
    # Determine output file path
    output_file = save_path if save_path else os.path.join(output_dir, "cutting_sequence_steps.png") if output_dir else "cutting_sequence_steps.png"
    
    # Number of paths
    num_paths = len(optimized_paths)
    
    # Determine grid layout
    import math
    if num_paths <= 4:
        rows, cols = max(1, num_paths // 2), min(2, num_paths)
    elif num_paths <= 9:
        rows, cols = math.ceil(math.sqrt(num_paths)), math.ceil(math.sqrt(num_paths))
    else:
        # Calculate best rectangular layout
        sqrt_paths = math.sqrt(num_paths)
        cols = math.ceil(sqrt_paths)
        rows = math.ceil(num_paths / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle different subplot configurations
    if num_paths == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        # Flatten 2D array to 1D list
        axes = axes.flatten() if hasattr(axes, 'flatten') else [ax for row in axes for ax in row]

    # Colors for paths
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Draw each step
    for step in range(num_paths):
        ax = axes[step]
        
        # Add background image
        if processed_image is not None:
            ax.imshow(processed_image, cmap='gray', alpha=0.3, origin='upper')
        
        # Draw previous paths in gray
        for prev_step in range(step):
            prev_path = optimized_paths[prev_step]
            if prev_path.points:
                prev_x = [p.x for p in prev_path.points]
                prev_y = [p.y for p in prev_path.points]
                ax.plot(prev_x, prev_y, 'gray', linewidth=1, alpha=0.5)
        
        # Draw current path
        current_path = optimized_paths[step]
        if current_path.points:
            # Extract coordinates
            curr_x = [p.x for p in current_path.points]
            curr_y = [p.y for p in current_path.points]
            
            color = colors[step % len(colors)]
            
            # Draw travel move if not first step
            if step > 0:
                prev_path = optimized_paths[step-1]
                if prev_path.points:
                    prev_end = prev_path.points[-1]
                    curr_start = current_path.points[0]
                    ax.plot([prev_end.x, curr_start.x], [prev_end.y, curr_start.y], 'k--', alpha=0.7, linewidth=2)
            
            # Draw cutting path and start point
            ax.plot(curr_x, curr_y, color=color, linewidth=3)
            ax.scatter(current_path.points[0].x, current_path.points[0].y, color='green', s=150, marker='o', edgecolor='black', linewidth=2)

        # Set title and formatting
        path_length = current_path.length()
        title = "#" + str(step+1) + ": " + str(int(path_length)) + "mm"
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(num_paths, rows * cols):
        axes[i].set_visible(False)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Saved:", output_file)
    plt.close()
