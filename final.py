import sys
sys.path.append('/usr/lib/python3/dist-packages')
import cv2 as cv
#from matplotlib import pyplot as plt
import numpy as np
#from pylab import *
#from scipy.ndimage import measurements
#import pytesseract

def findfloor(IMAGE, IMAGE2):
    eps = 0
    down = 0
    B = 0
    one = 0
    two = 0
    three = 0
    four=  0
    five = 0
    up = 0

    floor = 1
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
    return floor