import sys
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/usr/local/lib/python3.7/dist-packages')
#from cv2 import cv
from picamera.array import PiRGBArray
from picamera import PiCamera
from pymongo import MongoClient
import time
from final import findfloor
import MySQLdb
#import mysql.connector

db = MySQLdb.connect(host='127.0.0.1',    # your host, usually localhost
                     user='user',         # your username
                     password='citflow',  # your password
                     db='cameradb')        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like

# print all the first cell of all the rows
#for row in cur.fetchall():
    #print row[0]


camera = PiCamera()
camera.resolution = (640*2, 360*2)
#camera.stereo_mode = ("'side-by-side'")
rawCapture = PiRGBArray(camera, size=(640*2, 360*2))
time.sleep(0.1)

#try:
    #conn = MongoClient()
    #print("Connected successfully!")
#except:
    #print("Can't connect")
    
#db = conn.database
#collection = db.test_floors

prev = []
cur.execute("CREATE TABLE if not exists Camerafloors (floor VARCHAR(255));")

# result = []

#cycle through the stream of images from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array #the frame will be stroed in the varible called image
    if len(prev) > 0:
    #     result = image-prev
    # print(result)
        floor = findfloor(prev,image)
        print(floor)
        cur.execute("INSERT INTO Camerafloors (floor) VALUES (%s);", (str(floor),))
        db.commit()
        #collection.insert({"floor": floor})
        #print("inserted %s\n", floor)
    prev = image
    rawCapture.truncate(0)

db.close()
    
	#add your processing code here to process the image array
