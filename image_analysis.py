import sys
sys.path.append('/usr/local/lib/python3.7/dist-packages')

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from pylab import *
from scipy.ndimage import measurements

import pytesseract

from sklearn.cluster import MeanShift, estimate_bandwidth


# IMAGE = cv.imread("./Frames/frame" + str(i)+ ".jpg", cv.IMREAD_COLOR)
# IMAGE = cv.imread("./Images/IMG_0738.JPG")

# Find difference between images
# 1 & 2- 121, 124
# 2 & 3- 264, 274
# 3 & 4- 389, 391
# 4 & 5- 498 & 500 

# IMAGE = cv.imread("./Frames/frame16.jpg")
# IMAGE = cv.imread("./Images/IMG_1003.jpeg")
# IMAGE2 = cv.imread("./Frames/frame390.jpg")

# image_data = np.asarray(IMAGE, dtype=np.float32)
# image_data2 = np.asarray(IMAGE2, dtype=np.float32)
# diff = image_data2 - image_data


## Get clusters
# X = np.array(zip([i[0] for i in centers],np.zeros(len(centers))), dtype=np.int)
# bandwidth = estimate_bandwidth(X, quantile=0.1)
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_

# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)

# for k in range(n_clusters_):
#     my_members = labels == k
#     print ("cluster {0}: {1}".format(k, X[my_members, 0]))

# # Plot clusters
# centers = np.array(centers)
# x,y = centers.T
# plt.scatter(x,y)
# plt.show()


# # Hardcoded values
# floor = 1
# for i in range(551):
#     IMAGE = cv.imread("./Frames/frame" + str(i)+ ".jpg", cv.IMREAD_COLOR)
#     IMAGE2 = cv.imread("./Frames/frame" + str(i+1)+ ".jpg", cv.IMREAD_COLOR)

#     image_data = np.asarray(IMAGE, dtype=np.float32)
#     image_data2 = np.asarray(IMAGE2, dtype=np.float32)
#     diff = image_data2 - image_data

#     # Get rid of red
#     np.where(diff[:, :, 2] >= 0, diff[:, :, 2], 0)

#     # Find avg x value 
#     x = np.mean(np.where(diff[:, :, 2] >= 175), axis = 0)

#     xavg = np.mean(x)
#     biglist.append(x)

#     eps = 8
#     avg2 = 723
#     avg3 = 769
#     avg4 = 798
#     avg5 = 834

#     if xavg <= avg2-eps:
#         # print("on floor 1")
#         floor = 1
#     elif avg2-eps <= xavg <= avg3-eps:
#         # print("on floor 2")
#         floor = 2
#     elif avg3-eps <= xavg <= avg4-eps:
#         # print("on floor 3")
#         floor = 3
#     elif avg4-eps <= xavg <= avg5-eps:
#         # print("on floor 4")
#         floor = 4
#     elif avg4+eps <= xavg:
#         floor = 5
#         print("on floor 5")
#     print(i)
#     print("on floor %d\n" %floor)
#     print(x)

    # y = np.mean(np.where(diff[:, :, 2] >= 200), axis = 1)


# cv.imshow("diff", diff)
# cv.imwrite('./diff3&4half.png', diff)

# # Saliency method
# # saliency = cv.saliency.StaticSaliencyFineGrained_create()
# # (success, saliencyMap) = saliency.computeSaliency(IMAGE)
# # threshMap = cv.threshold(saliencyMap.astype("uint8"), 0, 255,
# # cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

# saliency = cv.saliency.StaticSaliencySpectralResidual_create()
# (success, saliencyMap) = saliency.computeSaliency(IMAGE)
# saliencyMap = (saliencyMap * 255).astype("uint8")
# cv.imwrite('./saliencymap', saliencyMap)
# cv.imshow("Image", IMAGE)
# cv.imshow("Output", saliencyMap)
# # cv.imshow("Thresh", threshMap)
# # cv.imwrite('./threshMap.png', threshMap)
# cv.waitKey(0)



# CONFIG = ("-l eng --oem 1 --psm 7")
# TEXT = pytesseract.image_to_string(IMAGE, lang='eng', config='config')
# print(TEXT)

## CHANGE CONTRAST
# alpha = 2.2 # contrast control
# beta = 50    # brightness control
# new_image = np.zeros(IMAGE.shape, IMAGE.dtype)
# new_image = np.clip(alpha*IMAGE + beta, 0, 255)
# cv.imshow('Original Image', IMAGE)
# cv.imshow('New Image', new_image)
# cv.imwrite('./bright_output1.png', new_image)

# cv.waitKey()

    # IMAGE2 = cv.imread("./Frames/frame" + str(i-1)+ ".jpg")

    # image_data = np.asarray(IMAGE, dtype=np.float32)
    # image_data2 = np.asarray(IMAGE2, dtype=np.float32)

   # if image_data[:,:,0] < 150 and image_data[:,:,1] < 150

    # imgtemp3 = image_data[:, :, 2] == 255
    # imgtemp = image_data[:, :, 1] < 230

    # imgtemp = (imgtemp2 == imgtemp3)
    # print(imgtemp)

    # imgtemp = np.abs(image_data[:, :, 2]- image_data2[:, :, 2])

    # lw, num = measurements.label(imgtemp)
    # area = measurements.sum(imgtemp, lw, index = np.arange(lw.max()+1))
    # maxind = np.argmax(lw)
    # col = maxind%4032
    # row = (maxind - col)/4032
    # plt.imshow(imgtemp*255)
    # plt.draw()
    # plt.pause(0.01)
    # plt.clf()

# plt.show()
