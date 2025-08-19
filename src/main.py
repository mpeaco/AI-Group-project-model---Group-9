from image_processing import *
import random
import argparse
import sys
import cv2 as cv
from potrace_wrapper import bitmap_to_vector
import pyvips
from rich import print, pretty
import time



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

    random_idx = random.randint(0, len(stored_images)) - 1
    if len(stored_images) == 0:
        return
    if random_image:
        return stored_images[random_idx]
    return stored_images[1]


def store_processed_image(storage_directory):
    """
    Store the processed images
    To be used to demo hwo the processing has worked
    Just for experimenting and testing
    """
    
    pass


def main():
    """
    Main function to run the program
    """
    pretty.install()
    
    parser = argparse.ArgumentParser(description="Process and extract lines from an image")
    parser.add_argument("--pyvips", '-p', action="store_true", help='use pyvips or default to opencv')
    args = parser.parse_args()
    
    use_pyvips = args.pyvips  # flag to set wether to use pyvips or opencv
    print("\nUsing pyvips: {}".format(use_pyvips))
    
    sample_images_path = "sample_data"
    processed_images_directory = "processed_images"

    # Load a sample image

    sample_image = load_sample_images(sample_images_path, use_pyvips=use_pyvips, random_image=False)

    print("Sample image loaded, type: {}".format(type(sample_image)))

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

    # perform processing of the image using open cv
    if not use_pyvips:
        o_start_time = time.perf_counter()
        processed_image = image_processing_opencv(sample_image)
        o_end_time = time.perf_counter()
        print("OpenCV processing time: {}".format(o_end_time - o_start_time))
    
    """
    The code below is used for opencv line extraction
    Not needed for using potrace and svg2path
    
    # Cannyedge detection
    #processed_edge_image = edge_detection(processed_image)
    # Line extraction using openCV
    #processed_lines = line_extraction(processed_edge_image)
    """

    if use_pyvips:
        p_start_time = time.perf_counter()
        processed_pyvips_image = image_processing_pyvips(pyvips_im)
        p_end_time = time.perf_counter()
        print("Pyvips image processing time: {}".format(p_end_time - p_start_time))
    if use_pyvips:
        cv.imshow("processed image", processed_pyvips_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        cv.imshow("processed image", processed_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Create a bitmap and then vector image

    if use_pyvips:
        file_path = create_bitmap(processed_pyvips_image)
        output_image_path = bitmap_to_vector(file_path)
    else:
        file_path = create_bitmap(processed_image)
        output_image_path = bitmap_to_vector(file_path)

    # Now extract the paths from the image
    paths = get_paths(output_image_path)
    points = set()
    edges = []
    for path in paths:
        #print(path.length)
        for segment in path:
            #print(segment)
            p1 = (segment.start.real, segment.start.imag)
            p2 = (segment.end.real, segment.end.imag)
            points.add(p1)
            points.add(p2)
            edges.append((p1, p2))
    
    #print(path)
    print("\nNumber of points: {}".format(len(points)))
    print("Number of edges: {}".format(len(edges)))
    print("\nImage processing complete")
    
    print(type(path))
    print(path)
    # Now the path needs to be processed and optimised
    
    

if __name__ == "__main__":
    main()
