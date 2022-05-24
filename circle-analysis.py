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

#1197x185
eps = 0
down = 0
B = 0
one = 0
two = 0
three = 0
four=  0
five = 0
up = 0

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

# centers = []

# for i in range(552):
#     IMAGE1 = cv.imread("./Frames/frame" + str(i)+ ".jpg", cv.IMREAD_COLOR)

#     pts = np.array([(365,321),(1526,494), (1551,661), (364,506)])
#     IMAGE = four_point_transform(IMAGE1, pts)

#     gray = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)
#     gray = cv.medianBlur(gray, 5)
#     rows = gray.shape[0]
#     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=20, maxRadius=1000)

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for j in circles[0, :]:
#             center = (j[0], j[1])
#             centers.append(center)
#             cv.circle(IMAGE, center, 1, (0, 100, 100), 3)
#             radius = j[2]
#             cv.circle(IMAGE, center, radius, (255, 0, 255), 3)
        # cv.imshow("detected circles", IMAGE)
    # cv.imwrite('./warped-circles.png', IMAGE)
# print(centers)


#Calibration
X = np.array(zip([i[0] for i in centers],np.zeros(len(centers))), dtype=np.int)
bandwidth = estimate_bandwidth(X, quantile=0.1)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
avglist = []
for k in range(n_clusters_):
    my_members = labels == k
    # print ("cluster {0}: {1}".format(k, X[my_members, 0]))
    xavg = np.mean(X[my_members, 0])
    avglist.append(xavg)

avglist.sort()
print(avglist)

down, B, one, two, three, four, five, up = avglist
eps = 10

print(one)
# centers = np.array(centers)
# x,y = centers.T
# plt.scatter(x,y)
# plt.show()
floor = 1
# for i in range(551):
    # IMAGE = cv.imread("./Frames/frame" + str(i)+ ".jpg", cv.IMREAD_COLOR)
    # IMAGE2 = cv.imread("./Frames/frame" + str(i+1)+ ".jpg", cv.IMREAD_COLOR)
IMAGE = cv.imread("./Frames/frame389.jpg", cv.IMREAD_COLOR)
IMAGE2 = cv.imread("./Frames/frame391.jpg", cv.IMREAD_COLOR)

pts = np.array([(365,321),(1526,494), (1551,661), (364,506)])
IMAGE = four_point_transform(IMAGE1, pts)
IMAGE2 = four_point_transform(IMAGE2, pts)

image_data = np.asarray(IMAGE, dtype=np.float32)
image_data2 = np.asarray(IMAGE2, dtype=np.float32)
diff = image_data2 - image_data

# Get rid of red
np.where(diff[:, :, 2] >= 0, diff[:, :, 2], 0)


# Find avg x value 
x = np.mean(np.where(diff[:, :, 2] >= 200), axis = 0)

xavg = np.mean(x)

if xavg <= two-eps:
    # print("on floor 1")
    floor = 1
elif two-eps <= xavg <= three-eps:
    # print("on floor 2")
    floor = 2
elif three-eps <= xavg <= four-eps:
    # print("on floor 3")
    floor = 3
elif four-eps <= xavg <= five-eps:
    # print("on floor 4")
    floor = 4
elif four+eps <= xavg:
    floor = 5
    # print("on floor 5")
# print(i)
print("on floor %d\n" %floor)
print (xavg)

cv.imwrite('./warped-diff.png', diff)
cv.imwrite('./warped1.png', IMAGE)
cv.imwrite('./warped2.png', IMAGE2)