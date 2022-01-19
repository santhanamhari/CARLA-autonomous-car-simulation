# Lane Detector Fixed 1.1


import setup_path
import airsim
import cv2
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

from homography_helper import *

# Choose Camera Type: Scene Camera Selected

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")

cameraType = "scene"

for arg in sys.argv[1:]:
  cameraType = arg.lower()


cameraTypeMap = { 
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}


if (not cameraType in cameraTypeMap):
  printUsage()
  sys.exit(0)


print (cameraTypeMap[cameraType])

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

help = False


fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print (textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime=time.clock()
fps = 0

#Setting the lane keep updates
x_left = -1
x_right = -1

time.sleep(10)
# ================================================================================================================================
# Main Method Loop
# ================================================================================================================================
while True:

    # =============================================================================================================================
    # Lane Detection
    # =============================================================================================================================

    car_state = client.getCarState()

    if (car_state.speed < 4.0):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0

    client.setCarControls(car_controls)

    # Get Image from Simulator
    rawImage = client.simGetImages([airsim.ImageRequest("0", cameraTypeMap[cameraType],False,False)])

    # Convert to RGB png image
    rawImage=rawImage[0]
    rawImage1d = np.fromstring(rawImage.image_data_uint8,dtype=np.uint8)
    rawImage_rgb = rawImage1d.reshape(rawImage.height,rawImage.width,3)

    img = rawImage_rgb

    #Apply Homography to convert front view to top view image

    birdseye = homography(img)

    # Detect and show Lanes
    i3, left, right = Lane_Detection_and_Overlay(birdseye)

    C = (left != -1 and right != -1)
    if C:
    	x_left = left
    	x_right = right
 
    cv2.putText(i3,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
 
    i3 = cv2.cvtColor(np.array(i3), cv2.COLOR_RGB2BGR)

    
    # Steering Controller

    car_state = client.getCarState()

    lane_center = (x_left + x_right) * 0.5
    #print('Lane Center = ',lane_center)

    # Heading of the car is the center of the image in x

    car_heading = 320 

    cntrl = lane_center - car_heading
    
    #print('Control Value = ',cntrl)

    # If Lane center is to the left of the vehicle heading, turn left to follow the lane

    if (cntrl < 0):
        car_controls.throttle = 0.2
        car_controls.steering = -0.2*np.abs(cntrl/100)
        #print('Turning Left!')

    # If Lane center is to the right of the vehicle heading, turn right to follow the lane
    
    elif (cntrl > 0):
        car_controls.throttle = 0.2
        car_controls.steering = 0.2*np.abs(cntrl/100)
        #print('Turning Right!')

    client.setCarControls(car_controls)

    cv2.imshow("Lane",i3)

    frameCount  = frameCount  + 1
    endTime=time.clock()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime

    key = cv2.waitKey(1) & 0xFF;

    if (key == 27 or key == ord('q') or key == ord('x')):
        break;

client.enableApiControl(False)