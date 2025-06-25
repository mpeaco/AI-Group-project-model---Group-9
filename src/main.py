from image_processing import *
import random
import argparse
import sys
import cv2 as cv
from potrace_wrapper import bitmap_to_vector
import pyvips



def load_sample_images(filepath: str, use_pyvips: bool = False):
    """
    Loads the sample images for testing
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

    print("images found: ", image_files)
    print(type(image_files[0]))

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
    
    return stored_images[3]


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

    use_pyvips = False  # flag to set wether to use pyvips or opencv

    sample_images_path = "sample_data"
    processed_images_directory = "processed_images"

    # Load a sample image

    sample_image = load_sample_images(sample_images_path, use_pyvips=use_pyvips)

    print("Sample image loaded, type: {}".format(type(sample_image)))

    pyvips_im = None
    if use_pyvips:
        print("It is a pyvips image")
        pyvips_im = sample_image  # Store the pyvips image as a new variable
        sample_image = convert_pyvips_to_numpy(
            sample_image
        )  # convert the image to a numpy array to display

    if sample_image.any():
        print("sample image type before imshow {}".format(type(sample_image)))
        cv.imshow("sample image", sample_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()

    # perform processing of the image using open cv
    processed_image = image_processing_opencv(sample_image)

    """
    The code below is used for opencv line extraction
    Not needed for using potrace and svg2path
    
    # Cannyedge detection
    #processed_edge_image = edge_detection(processed_image)
    # Line extraction using openCV
    #processed_lines = line_extraction(processed_edge_image)
    """

    if use_pyvips:
        processed_pyvips_image = image_processing_pyvips(pyvips_im)

    if use_pyvips:
        processed_pyvips_image = convert_pyvips_to_numpy(processed_pyvips_image)
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
        for segment in path:
            p1 = (segment.start.real, segment.start.imag)
            p2 = (segment.end.real, segment.end.imag)
            points.add(p1)
            points.add(p2)
            edges.append((p1, p2))
    
    print("Number of points: {}".format(len(points)))
    print("Number of edges: {}".format(len(edges)))
    


if __name__ == "__main__":
    main()
