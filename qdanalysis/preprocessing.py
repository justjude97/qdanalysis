"""
preprocessing.py

holds all preprocessing code for the different datasets
"""

import cv2 as cv
import numpy as np

def preprocess(image: np.ndarray):
    #2d array assumed to be grayscale image
    image_gs = image if len(image.shape) < 3 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #invert to make stroke pixels white (foreground). assumed to be an integer array.
    image_gs = np.iinfo(image_gs.dtype).max - image_gs

    k_size = [3, 3]
    #NOTE: may replace with a morphological operation
    #blurring only aplied to mask generation, not for feature extraction
    image_blurred = cv.GaussianBlur(image_gs, k_size, 0)

    image_bin = cv.threshold(image_blurred, 0, 1, cv.THRESH_OTSU)[1]

    return image_gs, image_bin