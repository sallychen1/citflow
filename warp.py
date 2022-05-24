import sys
sys.path.append('/usr/local/lib/python3.7/dist-packages')

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from pylab import *
from scipy.ndimage import measurements

import pytesseract

from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.filters import threshold_local
import imutils

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

image = cv.imread("./Frames/frame389.jpg", cv.IMREAD_COLOR)
#compute the ratio of the old height to the new height, clone it, and resize it
ratio = 1
orig = image.copy()
# image = imutils.resize(image, height = 185)

# convert the image to grayscale, blur it, and find edges in the image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 75, 200)

# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)
cnts = [cnts[33], cnts[35], cnts[38]]

minW = 239847927
minH = 2394872894
maxW = 0
maxH = 0
# loop over the contours
for c in cnts:
	# approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.01 * peri, True)
    for coord in approx:
        print(coord)
        print("end")
        if coord[0][0] < minW:
            minW = coord[0][0]
        if coord[0][0] > maxW:
            maxW = coord[0][0]
        if coord[0][1] < minH:
            minH = coord[0][1]
        if coord[0][1] > maxH:
            maxH = coord[0][1]
    # screenCnt = approx
    # print(approx)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    # if len(approx) == 4:
    #     screenCnt = approx
    #     break
# show the contour (outline) of the piece of paper
cv.drawContours(image, cnts, -1, (0, 255, 0), 2)
cv.imwrite("./imagecontour.png", image)
# print(cnts)

# apply the four point transform to obtain a top-down view of the original image
# warped = four_point_transform(orig, screenCnt.reshape(2, 2))
warped = four_point_transform(orig, np.array([[minW, minH], [maxW, minH], [maxW, maxH], [minW, maxH]]))

# # convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
# warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255
 
# show the original and scanned images
# cv.imwrite("./original.png", imutils.resize(orig, height = 650))
# cv.imwrite("./scanned.png", imutils.resize(warped, height = 650))

cv.imwrite("./original.png", orig)
cv.imwrite("./scanned.png", warped)
