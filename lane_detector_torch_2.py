# Robust Lane Detection implementation for AirSim

import argparse
from lane_detection_helper import *

# globel param
# dataset setting
img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2

# path
#train_path = "./data/train_index.txt"
#val_path = "./data/val_index.txt"
#test_path = "./data/test_index_demo.txt"
#save_path = "./save/result/"
pretrained_path='D:/Airsim/Robust-Lane-Detection-master/LaneDetectionCode/pretrained'

# weight
class_weight = [0.02, 1.02]

"""Dataloader
"""

from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
import pdb

"""Model"""

import torch
import config
# from config import args_setting
# from dataset import RoadSequenceDataset, RoadSequenceDatasetList
from model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import setup_path
import airsim
import cv2
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
import tempfile
import time


# OPEN CV SHOW

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

#===============================================================================
# Main Method 
#===============================================================================

args = args_setting()
torch.manual_seed(args.seed)
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#turn image into floatTensor
op_transforms = transforms.Compose([transforms.ToTensor() , transforms.Resize((128,256), interpolation=2)])

# load model and weights
model = generate_model(args)
#print('Model Success!')
class_weight = torch.Tensor(config.class_weight)
criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

pretrained_dict = torch.load(config.pretrained_path)
model_dict = model.state_dict()
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
model_dict.update(pretrained_dict_1)
model.load_state_dict(model_dict)
"""output_result(model, sample_batched, device)
print('Success!')
"""
rawImage_list=[]
a = time.time()
for i in range(5):
  
  rawImage = client.simGetImages([airsim.ImageRequest(0, cameraTypeMap[cameraType],False,False)])
  rawImage1d = np.fromstring(rawImage[0].image_data_uint8,dtype=np.uint8)
  rawImage_rgb = rawImage1d.reshape(rawImage[0].height,rawImage[0].width,3)

  rawImage_list.append(torch.unsqueeze(op_transforms(rawImage_rgb), dim=0))


# ================================================================================================================================
# Main Method Loop
# ================================================================================================================================
while True:

    car_state = client.getCarState()
                
    print('car speed: {0}'.format(car_state.speed))

    if (car_state.speed < 1.50):
      car_controls.throttle = 0.5
    else:
        car_controls.throttle = 0.0

    car_controls.steering = 0

    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))

    client.setCarControls(car_controls)

    # =============================================================================================================================
    # Lane Detection
    # =============================================================================================================================

    # Get Image from Simulator
    a = time.time()
    rawImage = client.simGetImages([airsim.ImageRequest(0, cameraTypeMap[cameraType],False,False)])
    rawImage1d = np.fromstring(rawImage[0].image_data_uint8,dtype=np.uint8)
    rawImage_rgb = rawImage1d.reshape(rawImage[0].height,rawImage[0].width,3)
    print(time.time()-a)

    rawImage_list.pop(0)  

    rawImage_list.append(torch.unsqueeze(op_transforms(rawImage_rgb), dim=0))
    

    data = torch.unsqueeze(torch.cat(rawImage_list, 0),dim=0)
    
    sample_batched = {'data': data}

    a = time.time()
    output = output_result(model, sample_batched, device)
    #print(np.array(output))
    #print(time.time()-a)

    #print(output.size)

    output_show = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

    cv2.putText(output_show,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
    
    cv2.imshow("Raw Image",output_show)


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
