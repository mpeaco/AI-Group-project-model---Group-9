from pathlib import Path
from pathlib import PosixPath
import pyvips
import matplotlib.pyplot as plt
from PIL import Image, ImageShow
import numpy as np
import cv2 as cv
import pyvips
import math
from skimage.morphology import skeletonize
from svgpathtools import svg2paths


def image_processing_pyvips(image):
    
    image = image.gaussblur(1.5)
    if image.hasalpha():
        image = image.flatten(background=[255]).unpremultiply()
    #print(image.get_scale())
    image = image.colourspace("b-w")
    if image.bands > 1:
        image = image.extract_band(0)
    
    #image = image < 128
    
    image = image.copy(interpretation="b-w")
    image = convert_pyvips_to_numpy(image)
    _, otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(otsu, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
    
    return cleaned


def image_processing_opencv(image):
    """
    Takes an image and converts to greyscale
    perform smoothing
    threshold

    """

    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 1.4)
    
    # Try Otsu's automatic thresholding for better results
    ret, img_thresholded = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(f"Otsu threshold value: {ret}")
    print(type(img_thresholded))
    
    # Use smaller kernel to preserve more detail
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(img_thresholded, cv.MORPH_OPEN, kernel)

# Fill small holes
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
    return cleaned


def edge_detection(img):
    canny_edges = cv.Canny(img, 100, 300)

    return canny_edges


def line_extraction(img):
    """
    Extracts contours using opencv findContours

    Args:
        img (ndarray): processed image

    Returns:
        list: contours found
    """
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("Contours found: {}", len(contours))
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 20:
            filtered_contours.append(contour)
            print(f"Kept contour with area: {area}")
    print("After filtering: {}".format(len(filtered_contours)))
    return filtered_contours


def create_bitmap(img):
    """
    Needs to create a bitmap from the processed image to save to root

    Args:
        img (numpy.ndarray): the processed image to be stored
    """
    output_filename = "processed_bitmap_image.pbm"
    cv.imwrite(output_filename, img)
    return output_filename


def get_paths(image_path):
    paths, _ = svg2paths(image_path)
    if paths:
        print("found paths")
        print("Number of paths found {}".format(len(paths)))
        print("Type of Paths: {}".format(type(paths[0])))
    return paths

def convert_pyvips_to_numpy(image):
    """
    Converts a pyvips.Image to a numpy array

    Args:
        image (pyvips.Image): image

    Returns:
        numpy.ndarray: numpy array of image
    """
    pyvips_im = image
    height = image.height
    width = image.width
    bands = image.bands
    print("It is a pyvips image")
    image = np.frombuffer(pyvips_im.write_to_memory(), dtype=np.uint8)
    np_image = image.reshape(height, width, bands)

    return np_image
