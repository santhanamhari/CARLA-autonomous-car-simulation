# CARLA-autonomous-car-simulation

## Introduction

We implement an autonomous car simulation in Carla, using deep learning and computer vision principles. Firstly, to track lanes on the road, we use Canny Edge detection and Robust lane detection. We track lanes based on the bird's eye view after applying a homography transformation with the front view. Then we use YOLO to ensure that we avoid pedestrians. Finally, we implement a basic controller to guide the car using the information from the lane detection and object detection modules. 

## Setup
CIS 581 Final Project: Team 10, Environment and Code Instructions
(This instruction is for windows)

1. Install AirSim, follow steps at : https://microsoft.github.io/AirSim/build_windows/

2. Install the relevant version of Visual Studio

3. Go to: https://github.com/microsoft/AirSim/releases and download the latest version of the City package for Windows

3. Download car and car2 folders from the following link: 
	Replace car folder with the downloaded folder.	

Instructions to run scripts: 
Download Models and Files Here: https://drive.google.com/file/d/1fUXZ58jPB7Mf-d8he7Peq2wdAeRkx6h_/view?usp=sharing

Watch Video here: https://drive.google.com/file/d/1oiLPbsf7IzoZcyqOO85P_p1nhx_jkJNl/view?usp=sharing

1. Execute CityEnviron - navigate vehicle manually to location shown in demo video
2. Open Developer Command Prompt for VS 2019
3. Navigate to the correct drive
4. Navigate to Airsim/venv/Scripts
5. Execute the activate file

You are now in virtual environment.

1. Navigate to Airsim/PythonClient/car

Command to run traditional lane detection:
python classic_lane_detection_project_demo.py


Command to run Robust Lane Net:
python lane_detector_torch_2.py

2. Navigate to Airsim/PythonClient/car2

Command to run Object Detection with YOLO:
python yolov3_final_demo

