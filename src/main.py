from image_processing import *
import argparse
import cv2 as cv
from vector_utils import bitmap_to_vector as cv_bitmap_to_vector
from dxf_utils import svg_to_dxf
from path_optimisation import (process_path, optimize_cutting_sequence, generate_cutting_report, 
                              PathOptimizer, visualize_paths, create_cutting_sequence_animation)
import pyvips
from rich import print, pretty
import time
import os
from datetime import datetime



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
    image_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(image_directory.glob(ext))

    if len(image_files) == 0:
        return

    #print("\nImages found: ", image_files)
    #print(type(image_files[0]))

    stored_images = []

    if use_pyvips:
        for im in image_files:
            image = pyvips.Image.new_from_file(im)
            stored_images.append(image)

    else:
        for im in image_files:
            image = cv.imread(im)
            stored_images.append(image)

    # Use fixed image selection instead of random
    if len(stored_images) == 0:
        return
    if random_image:
        return stored_images[0]  # Return first image instead of random
    return stored_images[1] if len(stored_images) > 1 else stored_images[0]



def main():
    """
    Main function to run the program
    """
    pretty.install()
    
    parser = argparse.ArgumentParser(description="Process and extract lines from an image")
    parser.add_argument("--pyvips", '-p', action="store_true", help='use pyvips or default to opencv')
    args = parser.parse_args()
    
    use_pyvips = args.pyvips
    print(f"\nUsing pyvips: {use_pyvips}")
    
    sample_images_path = "sample_data"

    sample_image = load_sample_images(sample_images_path, use_pyvips=use_pyvips, random_image=False)
    print(f"Sample image loaded, type: {type(sample_image)}")

    pyvips_im = None
    if use_pyvips:
        pyvips_im = sample_image  # Store the pyvips image as a new variable
        sample_image = convert_pyvips_to_numpy(
            sample_image
        )  # convert the image to a numpy array to display

    if sample_image.any():
        cv.imshow("sample image", sample_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Perform image processing
    if not use_pyvips:
        o_start_time = time.perf_counter()
        processed_image = image_processing_opencv(sample_image)
        o_end_time = time.perf_counter()
        print(f"OpenCV processing time: {o_end_time - o_start_time:.4f}s")
    else:
        p_start_time = time.perf_counter()
        processed_pyvips_image = image_processing_pyvips(pyvips_im)
        p_end_time = time.perf_counter()
        print(f"Pyvips processing time: {p_end_time - p_start_time:.4f}s")
    # Display processed image
    if use_pyvips:
        cv.imshow("processed image", processed_pyvips_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # Create bitmap and vector
        file_path = create_bitmap(processed_pyvips_image)
        output_image_path = cv_bitmap_to_vector(file_path)
    else:
        cv.imshow("processed image", processed_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # Create bitmap and vector  
        file_path = create_bitmap(processed_image)
        output_image_path = cv_bitmap_to_vector(file_path)

    # Now extract the paths from the image
    paths = get_paths(output_image_path)
    
    print(f"\n=== PATH OPTIMIZATION ANALYSIS ===")
    print(f"Found {len(paths)} paths from image")
    
    # Convert SVG paths to CuttingPath objects for optimization
    cutting_paths = []
    for i, path in enumerate(paths):
        try:
            cutting_path = process_path(path)
            cutting_paths.append(cutting_path)
            print(f"Path {i+1}: {len(cutting_path.points)} points, length: {cutting_path.length():.2f}mm")
        except Exception as e:
            print(f"Error processing path {i+1}: {e}")
            continue
    
    if cutting_paths:
        print(f"\nSuccessfully converted {len(cutting_paths)} paths for optimization")
        
        # Generate report before optimization
        print("\n=== BEFORE OPTIMIZATION ===")
        original_report = generate_cutting_report(cutting_paths)
        print(f"Total paths: {original_report['path_count']}")
        print(f"Total cutting length: {original_report['total_cutting_length']:.2f}mm")
        print(f"Estimated cutting time: {original_report['estimated_time_minutes']:.2f} minutes")
        print(f"Pierce operations: {original_report['pierce_count']}")
        
        # Optimize cutting sequence using nearest neighbor TSP
        print("\n=== OPTIMIZING PATHS ===")
        optimization_start = time.perf_counter()
        optimized_paths = optimize_cutting_sequence(cutting_paths, method="nearest_neighbor")
        optimization_end = time.perf_counter()
        
        print(f"Path optimization completed in {optimization_end - optimization_start:.4f} seconds")
        
        # Generate report after optimization
        print("\n=== AFTER OPTIMIZATION ===")
        optimized_report = generate_cutting_report(optimized_paths)
        print(f"Total paths: {optimized_report['path_count']}")
        print(f"Total cutting length: {optimized_report['total_cutting_length']:.2f}mm")
        print(f"Estimated cutting time: {optimized_report['estimated_time_minutes']:.2f} minutes")
        print(f"Pierce operations: {optimized_report['pierce_count']}")
        
        # Calculate optimization improvements
        time_savings = original_report['estimated_time_minutes'] - optimized_report['estimated_time_minutes']
        if time_savings > 0:
            savings_percentage = (time_savings / original_report['estimated_time_minutes']) * 100
            print(f"\n✅ Optimization Results:")
            print(f"   Time saved: {time_savings:.2f} minutes ({savings_percentage:.1f}%)")
        else:
            print(f"\n⚠️ No time savings achieved (paths may already be optimal)")
        
        # Add lead-in/out to optimized paths
        print("\n=== ADDING LEAD-IN/OUT PATHS ===")
        optimizer = PathOptimizer()
        final_paths = []
        for path in optimized_paths:
            enhanced_path = optimizer.add_lead_in_out(path)
            final_paths.append(enhanced_path)
        
        print(f"Added lead-in/out to {len(final_paths)} paths")
        
        # VISUALIZATION: Show processed image and paths
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # Create timestamped folder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join("sample_data", "path_optimised_sample", f"run_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            # Load the processed image for visualization
            if use_pyvips:
                processed_img = processed_pyvips_image
            else:
                processed_img = processed_image
            
            # Create path visualizations with timestamped filenames
            comparison_path = os.path.join(output_folder, "path_optimization_comparison.png")
            sequence_path = os.path.join(output_folder, "cutting_sequence_steps.png")
            
            print("Creating path comparison visualization...")
            visualize_paths(cutting_paths, optimized_paths, processed_img, 
                          save_path=comparison_path)
            
            print("Creating cutting sequence animation...")
            create_cutting_sequence_animation(optimized_paths, processed_img,
                                            save_path=sequence_path)
            
            print(f"✅ Visualizations saved to: {output_folder}")
            
        except Exception as e:
            print(f"Visualization error: {e}")
            # Fallback to simple visualization without processed image
            print("Creating simplified path visualization...")
            simple_path = os.path.join(output_folder, "path_optimization_simple.png")
            visualize_paths(cutting_paths, optimized_paths, None, 
                          save_path=simple_path)
        
        # Display detailed path information
        print(f"\n=== DETAILED PATH ANALYSIS ===")
        total_travel_distance = 0
        for i, path in enumerate(final_paths):
            print(f"Path {i+1}:")
            print(f"  Points: {len(path.points)}")
            print(f"  Length: {path.length():.2f}mm")
            print(f"  Closed: {path.is_closed}")
            print(f"  Priority: {path.priority}")
            
            if i > 0:
                # Calculate travel distance from previous path
                prev_end = final_paths[i-1].points[-1]
                curr_start = path.points[0]
                travel_dist = prev_end.distance_to(curr_start)
                total_travel_distance += travel_dist
                print(f"  Travel from prev: {travel_dist:.2f}mm")
        
        print(f"\nTotal rapid travel distance: {total_travel_distance:.2f}mm")
        
    else:
        print("⚠️ No valid paths found for optimization")
    
    # Convert SVG to DXF for laser cutting
    print("\n=== DXF CONVERSION ===")
    dxf_output_path = svg_to_dxf(output_image_path)
    print(f"DXF file created: {dxf_output_path}")
    
    print(f"\n🎉 Processing complete! Files generated:")
    print(f"   - Bitmap: processed_bitmap_image.pbm")
    print(f"   - Vector: {output_image_path}")
    print(f"   - DXF: {dxf_output_path}")
    if 'output_folder' in locals():
        print(f"   - Optimizations folder: {output_folder}")
        print(f"     • Path comparison: path_optimization_comparison.png")
        print(f"     • Cutting sequence: cutting_sequence_steps.png")
    
if __name__ == "__main__":
    main()
