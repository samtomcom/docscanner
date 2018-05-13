# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from fpdf import FPDF
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True,
#	help = "Path to the image to be scanned")
#args = vars(ap.parse_args())

class ImageToPDF():
    """Convert a set of images into a pdf of one image per page"""

    def __init__(self, image):
        """Run all functions to produce pdf"""
        warped = self.format_image(image)
        self.save_png(warped)
        self.save_pdf('output/output.png')

    def format_image(self, image): 
        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        orig = cv2.imread(image)
        #ratio = orig.shape[0] / 500.0
        #image = imutils.resize(orig, height = 500)

        gray = self.grayscale(image)
        blurred = self.blur(gray)
        edged = cv2.Canny(blurred, 75, 200)

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        screenCnt = self.contour(edged)

        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = self.grayscale(warped)
        warped = self.threshold(warped)

        return warped

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def threshold(self, image):
        T = threshold_local(image, 11, offset = 10, method = "gaussian")
        return (image > T).astype("uint8") * 255

    def contour(self, image):
        cnts = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        return screenCnt

    def save_png(self, image):
        # write image to file
        cv2.imwrite('output/output.png', image)

    def save_pdf(self, image):
        pdf = FPDF()
        pdf.add_page()
        pdf.image(image, 1, 1, 208, 295)
        pdf.output('output/output.pdf', 'F')

conv = ImageToPDF('images/receipt.jpg')
