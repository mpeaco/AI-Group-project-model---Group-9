import cv2 as cv
import numpy as np

def bitmap_to_vector(bitmap_path, output_path=None):
    """
    Convert bitmap to vector using contours instead of potrace
    
    Args:
        bitmap_path (str): Path to bitmap image
        output_path (str, optional): Path for output SVG file
    Returns:
        str: Path to output SVG file
    """
    # Read bitmap
    img = cv.imread(bitmap_path, cv.IMREAD_GRAYSCALE)
    
    # Find contours - Use RETR_TREE to get ALL contours, not just external
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove noise
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 50:  # Only keep contours larger than 50 pixels
            filtered_contours.append(contour)
    
    print(f"Found {len(contours)} contours, kept {len(filtered_contours)} after filtering")
    contours = filtered_contours
    
    # Create SVG content
    svg_content = ['<?xml version="1.0" encoding="UTF-8" ?>\n']
    svg_content.append(f'<svg width="{img.shape[1]}" height="{img.shape[0]}" xmlns="http://www.w3.org/2000/svg">\n')
    
    # Convert contours to SVG paths
    for contour in contours:
        path_data = "M "
        for point in contour:
            x, y = point[0]
            path_data += f"{x},{y} "
        path_data += "Z"
        svg_content.append(f'  <path d="{path_data}" fill="none" stroke="black"/>\n')
    
    svg_content.append('</svg>')
    
    # Write SVG file
    if output_path is None:
        output_path = bitmap_path.replace('.pbm', '.svg')
        
    # Make sure the directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.writelines(svg_content)
    
    return output_path