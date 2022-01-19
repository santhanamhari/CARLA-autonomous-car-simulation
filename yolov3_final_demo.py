from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

#=================================================
#AIRSIM IMPORTS
#=================================================

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

# Load model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Darknet("config/yolov3.cfg", img_size=416).to(device)

model.load_darknet_weights("weights/yolov3.weights")

model.eval()  # Set in evaluation mode

classes = load_classes("data/coco.names")  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# Choose Camera Type to use: Scene Camera Selected

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

plt.figure(0)
fig, ax = plt.subplots(1)

#===============================================================================
# Main Method 
#===============================================================================

# ================================================================================================================================
# Main Method Loop
# ================================================================================================================================
while True:

    # =============================================================================================================================
    # Object Detection
    # =============================================================================================================================

    # Get Image from Simulator and Convert to RGB
    
    rawImage = client.simGetImages([airsim.ImageRequest(0, cameraTypeMap[cameraType],False,False)])
    rawImage1d = np.fromstring(rawImage[0].image_data_uint8,dtype=np.uint8)
    rawImage_rgb = rawImage1d.reshape(rawImage[0].height,rawImage[0].width,3)

    img = rawImage_rgb

    img = transforms.ToTensor()(img)

    img = F.interpolate(img.unsqueeze(0), size=(416,416), mode="nearest").to(device)

    with torch.no_grad():
      detections = model(img)
      detections = non_max_suppression(detections, 0.8, 0.4)
    
    ax.imshow(rawImage_rgb)

# Draw bounding boxes and labels of detections
    if detections[0] is not None:
        # Rescale boxes to original image
        detections = detections[0]
        #print(detections)
        detections = rescale_boxes(detections, 416, rawImage_rgb.shape[:2])
        #print(detections)
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.pause(0.01)
    ax.clear()

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
