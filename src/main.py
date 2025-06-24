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
    return stored_images[random_idx]


def store_processed_image():
    """
    Store the processed images
    To be used to demo hwo the processing has worked
    Just for experimenting and testing
    """
    pass


def main():
    use_pyvips = False
    sample_images_path = "sample_data"
    processed_images_directory = "processed_images"

    sample_image = load_sample_images(sample_images_path, use_pyvips=use_pyvips)
    print(type(sample_image))
    pyvips_im = None
    if use_pyvips:
        print("It is a pyvips image")
        pyvips_im = sample_image
        sample_image = convert_pyvips_to_numpy(sample_image)

    if sample_image.any():
        print("sample image type before imshow {}".format(type(sample_image)))
        cv.imshow("sample image", sample_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()

    # perform processing of the image
    processed_image = image_processing_opencv(sample_image)
    processed_edge_image = edge_detection(processed_image)
    processed_lines = line_extraction(processed_edge_image)

    if use_pyvips:
        processed_pyvips_image = image_processing_pyvips(pyvips_im)

    if processed_edge_image.any():
        cv.imshow("processed image", processed_edge_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()

    if use_pyvips:
        processed_pyvips_image = convert_pyvips_to_numpy(processed_pyvips_image)
        cv.imshow("processed image", processed_pyvips_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        cv.imshow("processed image", processed_edge_image)  # View a the sample image
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Create a bitmap and then vector image

    if use_pyvips:
        file_path = create_bitmap(processed_pyvips_image)
    else:
        file_path = create_bitmap(processed_edge_image)
    if processed_image.any():
        print("Image processed")
        output_image_path = bitmap_to_vector(file_path)

    get_paths(output_image_path)


if __name__ == "__main__":
    main()
