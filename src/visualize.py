import matplotlib.pyplot as plt
import numpy as np
import os
from path_optimisation import CuttingPoint


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
