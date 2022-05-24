import sys
sys.path.append('/usr/local/lib/python3.7/dist-packages')

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from pylab import *
from scipy.ndimage import measurements

# plt.axis([0,600,0,600])
# plt.ion()
# for i in range(1,552):

# IMAGE = cv.imread("./Frames/frame" + str(i)+ ".jpg", cv.IMREAD_COLOR)
# IMAGE = cv.imread("./Images/IMG_0738.JPG")

# alpha = 2.2 # Simple contrast control
# beta = 50    # Simple brightness control
# new_image = np.zeros(IMAGE.shape, IMAGE.dtype)
# new_image = np.clip(alpha*IMAGE + beta, 0, 255)

# cv.imshow('Original Image', IMAGE)
# cv.imshow('New Image', new_image)
# cv.imwrite('./bright_output1.png', new_image)
# Wait until user press some key
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

## HISTOGRAMS
#hist = cv.calcHist([image], [2], None, [256], [0.256])
#plt.plot(hist, color='r')
#plt.xlim([0, 256])
#plt.show()

# for i, col in enumerate(['b', 'g', 'r']):

# hist = cv.calcHist([image], [2], None, [256], [0,256])
# plt.plot(hist, "r")
# plt.xlim([0,256])
# plt.show()

# for i in range(len(image_data)):
#     for j in range(len(image_data[0])):
#         if image_data[i][j][2] > 240:
#             print(image_data[i][j])

## ANIMATED PLOT
# imgtemp = np.zeros((len(image_data), len(image_data[0])))
# plt.ion()
    # for k in range(253,256):

        # for i in range(len(image_data)):
        #     for j in range(len(image_data[0])):
                # if image_data[i][j][2] == k:
                #     imgtemp[i][j] = 1 
                # else:
                #     imgtemp[i][j] = 0

        # imgtemp = image_data[:, :, 2] == 255
        # print(len(imgtemp))  
        # print(imgtemp[0])
        # print(imgtemp[0][0])
        # new_img_temp = imgtemp.reshape()
        # lw, num = measurements.label(imgtemp)
        # #print(lw)
        # area = measurements.sum(imgtemp,lw,index = np.arange(lw.max()+1))

        # #print(area)
        # # print(max(area))
        # maxind = np.argmax(lw)
        # col = maxind%4032
        # row = (maxind - col)/4032

        # # ind = np.unravel_index(np.argmax(lw,axis=None),lw.shape)
        # # result = np.where(lw == np.amax(lw))
        # # print(ind)
        # print(row, col)
    # plt.imshow(imgtemp*255)
    # plt.draw()
    # plt.pause(1.5)
    # plt.clf()

# ORIG = IMAGE.copy()
# GRAY = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)

# GRAY = cv.GaussianBlur(GRAY, (21, 21), 0)
# (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(GRAY)
# IMAGE = ORIG.copy()
# cv.circle(IMAGE, maxLoc, 21, (255, 0, 0), 2)
# cv.imshow("Robust", IMAGE)
# cv.waitKey(0)

# (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
# cv.circle(image, maxLoc, 5, (255, 0, 0), 2)
# cv.imshow("Naive", image)
# cv.waitKey(0)


#print indices of certain color value
# indices = np.where(IMAGE == [125])
# print (indices)
# coordinates = zip(indices[0], indices[1])
# print (coordinates)

#for loop 