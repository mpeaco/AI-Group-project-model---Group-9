import os
path = os.path.abspath("vips-dev-8.17/bin")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path
path = os.path.abspath("potrace-1.16.win64")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path
from processing.image_processing import *
import cv2 as cv
from processing.vector_utils import bitmap_to_vector as cv_bitmap_to_vector
from utils.dxf_utils import svg_to_dxf
from processing.path_optimisation import (process_path, optimize_cutting_sequence, PathOptimizer)
from visualize import visualize_paths, create_cutting_sequence_animation
import pyvips
import time
from datetime import datetime
from pathlib import Path


# Loads images from a folder
def load_sample_images(filepath: str, use_pyvips: bool = False, random_image=False):
    """
    Loads the sample images for testing
    Real application will take one image. 
    Returns a random image to see the result

    Args:
        filepath (str): file path for images
        use_pyvips (bool, optional): Use pyvips for processing or not. Defaults to False.

    Returns:
        pyvips.Image or cv.image: random image from sample images
    """

    image_directory = Path(filepath)
    
    # What file types to look for
    image_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []

    # Find all image files
    for ext in image_extensions:
        image_files.extend(image_directory.glob(ext))

    # Check if we found any
    if len(image_files) == 0:
        return

    #print("\nImages found: ", image_files)
    #print(type(image_files[0]))

    stored_images = []

    # Load images with either pyvips or opencv
    if use_pyvips:
        # pyvips way
        for im in image_files:
            image = pyvips.Image.new_from_file(im)
            stored_images.append(image)
    else:
        # opencv way
        for im in image_files:
            image = cv.imread(im)
            stored_images.append(image)

    # Make sure we have images
    if len(stored_images) == 0:
        return
        
    # Return first image instead of random
    if random_image:
        return stored_images[0]
        
    # Return the second image if possible, otherwise first one
    if len(stored_images) > 1:
        return stored_images[1]
    else:
        return stored_images[0]


# Main function
def main():
    # Ask user if they want to use pyvips
    user_choice = input("Use pyvips? (y/n): ")
    use_pyvips = True if user_choice.lower() == 'y' else False
    print("Using pyvips:", use_pyvips)
    
    # Make a folder with timestamp for output in sample_results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"sample_data/sample_results/output_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print("Output folder:", output_folder)
    
    # Try to load sample images from folder
    sample_images_path = "sample_data"

    # Load an image
    sample_image = load_sample_images(sample_images_path, use_pyvips=use_pyvips, random_image=False)
    print("Image loaded:", type(sample_image))

    # Convert pyvips image to numpy if needed
    pyvips_im = None
    if use_pyvips:
        # Need to save original for processing
        pyvips_im = sample_image  
        # Convert to numpy to display
        sample_image = convert_pyvips_to_numpy(sample_image)  

    # Show original image
    if sample_image.any():
        print("Original image loaded successfully:", sample_image.shape)

    # Process the image - different ways depending on library
    if not use_pyvips:
        # OpenCV way
        o_start_time = time.perf_counter()
        processed_image = image_processing_opencv(sample_image)
        o_end_time = time.perf_counter()
        print("OpenCV time:", round(o_end_time - o_start_time, 4), "s")
    else:
        # PyVips way 
        p_start_time = time.perf_counter()
        processed_pyvips_image = image_processing_pyvips(pyvips_im)
        p_end_time = time.perf_counter()
        print("Pyvips time:", round(p_end_time - p_start_time, 4), "s")
        
    # Show what we got
    if use_pyvips:
        print("PyVips processed image shape:", processed_pyvips_image.shape)
        
        # Save files for pyvips version
        bitmap_filename = os.path.join(output_folder, "processed_bitmap_image.pbm")
        file_path = create_bitmap(processed_pyvips_image, bitmap_filename)
        svg_filename = os.path.join(output_folder, "processed_vector_image.svg")
        output_image_path = cv_bitmap_to_vector(file_path, svg_filename)
    else:
        print("OpenCV processed image shape:", processed_image.shape)
        
        # Save files for opencv version
        bitmap_filename = os.path.join(output_folder, "processed_bitmap_image.pbm")
        file_path = create_bitmap(processed_image, bitmap_filename)
        svg_filename = os.path.join(output_folder, "processed_vector_image.svg")
        output_image_path = cv_bitmap_to_vector(file_path, svg_filename)

    # Get paths from the SVG
    paths = get_paths(output_image_path)
    
    # Start path optimization
    print("\nPath Analysis:")
    print("Found", len(paths), "paths")
    
    # Need to convert SVG paths to our format
    cutting_paths = []
    for i, path in enumerate(paths):
        # Sometimes paths have errors
        try:
            cutting_path = process_path(path)
            cutting_paths.append(cutting_path)
            print("Path", i+1, ":", len(cutting_path.points), "points,", round(cutting_path.length(), 2), "mm")
        except Exception as e:
            # Skip bad paths
        # Skip bad paths
            print("Error processing path", i+1, ":", e)
            continue
    
    # Make sure we got some paths
    if cutting_paths:
        print("\nOptimizing", len(cutting_paths), "paths")
        
        # Check how things look before optimization
        print("\nBefore:")
        print("Paths:", len(cutting_paths))
        total_length = sum(path.length() for path in cutting_paths)
        print("Length:", round(total_length, 2), "mm")
        print("Pierces:", len(cutting_paths))
        
        # Try to optimize the path order
        print("\nOptimizing...")
        # Time how long it takes
        optimization_start = time.perf_counter()
        # Use nearest neighbor method
        optimized_paths = optimize_cutting_sequence(cutting_paths, method="nearest_neighbor")
        optimization_end = time.perf_counter()
        
        print("Done in", round(optimization_end - optimization_start, 4), "seconds")
        
        # Check how things look after optimization
        print("\nAfter:")
        print("Paths:", len(optimized_paths))
        optimized_length = sum(path.length() for path in optimized_paths)
        print("Length:", round(optimized_length, 2), "mm")
        print("Pierces:", len(optimized_paths))
        
        # See if we actually improved anything
        if total_length > optimized_length:
            length_savings = total_length - optimized_length
            savings_percentage = (length_savings / total_length) * 100
            print("\nSavings:")
            print("Length saved:", round(length_savings, 2), "mm (", round(savings_percentage, 1), "%)")
        else:
            print("\nPath order optimized")
            
        # Try to add lead-in/out paths - sometimes helpful for laser cutting
        print("\nAdding lead-in/out...")
        optimizer = PathOptimizer()
        
        # This is new - just added these
        final_paths = []
        for path in optimized_paths:
            # I had to add this method to PathOptimizer
            enhanced_path = optimizer.add_lead_in_out(path)
            final_paths.append(enhanced_path)
        
        print("Added to", len(final_paths), "paths")
        
        # Make some pictures to see what we did
        print("\nMaking visualizations...")
        
        # Need a special folder for the pictures
        vis_folder = os.path.join(output_folder, "visualizations")
        os.makedirs(vis_folder, exist_ok=True)
        
        # Sometimes this fails because of memory issues
        try:
            # Get processed image based on which method we used
            if use_pyvips:
                processed_img = processed_pyvips_image
            else:
                processed_img = processed_image
            
            # File paths for our images
            comparison_path = os.path.join(vis_folder, "path_optimization_comparison.png")
            sequence_path = os.path.join(vis_folder, "cutting_sequence_steps.png")
            
            # Make the visualizations
            print("Path comparison...")
            visualize_paths(cutting_paths, optimized_paths, processed_img, 
                          save_path=comparison_path)
            
            print("Cutting sequence...")
            create_cutting_sequence_animation(optimized_paths, processed_img,
                                            save_path=sequence_path)
            
            print("Saved to:", vis_folder)
            
        except Exception as e:
            # Handle visualization errors
            print("Visualization error:", e)
            print("Trying simpler version...")
            simple_path = os.path.join(vis_folder, "path_optimization_simple.png")
            # Try without the image background
            visualize_paths(cutting_paths, optimized_paths, None, 
                          save_path=simple_path)
        
        # Look at each path in detail
        print("\nPath Details:")
        # Calculate the travel distance between paths
        total_travel_distance = 0
        
        # Go through each path
        for i, path in enumerate(final_paths):
            # Print info about this path
            print("Path", i+1, ":")
            print("  Points:", len(path.points))
            print("  Length:", round(path.length(), 2), "mm")
            print("  Closed:", path.is_closed)
            print("  Priority:", path.priority)
            
            # Calculate travel distance (not for first path)
            if i > 0:
                # Need to go from end of previous to start of this one
                prev_end = final_paths[i-1].points[-1]
                curr_start = path.points[0]
                # This is the "rapid travel" distance
                travel_dist = prev_end.distance_to(curr_start)
                total_travel_distance += travel_dist
                print("  Travel from prev:", round(travel_dist, 2), "mm")
        
        # Show total travel
        print("\nTotal travel:", round(total_travel_distance, 2), "mm")
        
        # *** NEW: MATERIAL AND DEPTH MANAGEMENT ***
        print("\n==================================================")
        print("LASER CUTTING SETUP")
        print("==================================================")
        
        # Import the workflow functions
        from materials.workflow import run_workflow
        
        # Run interactive material and depth selection
        depth_manager = run_workflow(final_paths, output_folder)
        
        if depth_manager is None:
            print("Setup cancelled.")
            depth_manager = None  # Will use defaults later
        else:
            print("Setup complete!")
            
            # Show final cutting sequence
            cutting_sequence = depth_manager.get_cutting_sequence()
            print(f"\nCutting sequence: {len(cutting_sequence)} operations")
        
        print("==================================================")
        
    else:
        # No paths to optimize
        print("No paths found!")
    
    # Convert to DXF format for the laser cutter software
    print("\nDXF Conversion:")
    # Create the output file path
    dxf_output_path = os.path.join(output_folder, "processed_image.dxf")
    # 
    dxf_output_path = svg_to_dxf(output_image_path, dxf_output_path)
    print("DXF created:", dxf_output_path)
    
    # All done! Summarize what we made
    print("\nDone! Files in:", output_folder)
    print("  - Bitmap:", os.path.basename(bitmap_filename))
    print("  - Vector:", os.path.basename(svg_filename))
    print("  - DXF:", os.path.basename(dxf_output_path))
    
# Run the main function if this is the main script
if __name__ == "__main__":
    main()
