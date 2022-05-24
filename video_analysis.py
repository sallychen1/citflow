import sys
sys.path.append('/usr/local/lib/python3.7/dist-packages')

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

vid = cv.VideoCapture("Videos/IMG_0987.MOV")

#Extract frames
count = 0
success = 1

  
while success: 
    success, image = vid.read() 
    cv.imwrite("frame%d.jpg" % count, image)
    count += 1
