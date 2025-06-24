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
    image = image.median(3)
    if image.hasalpha():
        image = image.flatten(background=[255])
    print(image.get_scale())
    image = image.colourspace("b-w")
    image = image < 127
    if image.bands > 1:
        image = image.extract_band(0)
    image = image.copy(interpretation="b-w")

    return image


def image_processing_opencv(image):
    """
    Takes an image and converts to greyscale
    perform smoothing
    threshold

    """

    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 1.4)
    ret, img_thresholded = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
    print(type(img_thresholded))
    return img_thresholded


def edge_detection(img):
    canny_edges = cv.Canny(img, 100, 300)

    return canny_edges


def line_extraction(img):
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


def convert_pyvips_to_numpy(image):

    pyvips_im = image
    height = image.height
    width = image.width
    bands = image.bands
    print("It is a pyvips image")
    image = np.frombuffer(pyvips_im.write_to_memory(), dtype=np.uint8)
    np_image = image.reshape(height, width, bands)

    return np_image
