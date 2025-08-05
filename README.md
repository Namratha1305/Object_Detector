**Real-Time Object Detector with YOLOv4 and OpenCV**
This project is a real-time object detection application that uses your computer's webcam to identify objects in the video feed. It leverages the power of OpenCV and the YOLOv4-tiny model to draw bounding boxes and labels around detected objects.

<img width="478" height="539" alt="Detection" src="https://github.com/user-attachments/assets/d244c4f5-3dca-4215-b13d-a1f2165601ce" />
<img width="231" height="233" alt="image" src="https://github.com/user-attachments/assets/a7915776-6301-4415-9ef7-5f1c5f4bdaeb" />


Features
Real-Time Detection: Identifies objects instantly from a live webcam feed.

High Performance: Uses YOLOv4-tiny, a model optimized for speed and efficiency on standard hardware.

80 Object Classes: Capable of recognizing 80 different common objects from the COCO dataset (e.g., person, car, bottle, cell phone, cat, dog).

Simple & Extensible: Built with Python and OpenCV, making it easy to understand and modify.

Technology Stack
Python 3.x

OpenCV (Open Source Computer Vision Library)

NumPy

YOLOv4-tiny (You Only Look Once, version 4-tiny)

Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
Make sure you have Python 3 installed. Then, install the necessary Python libraries using pip:

pip install opencv-python
pip install numpy

2. Download YOLOv4-tiny Model Files
You need to download the model configuration, weights, and class names. Place these three files in the root directory of your project.

Configuration File: yolov4-tiny.cfg

Download Link: yolov4-tiny.cfg

Instructions: Right-click the link -> "Save Link As..." -> Set "Save as type" to "All Files" -> Save as yolov4-tiny.cfg.

Weights File: yolov4-tiny.weights

Download Link: yolov4-tiny.weights

Instructions: Click the link to download the ~23 MB file directly.

Class Names File: coco.names

Download Link: coco.names

Instructions: Right-click the link -> "Save Link As..." -> Set "Save as type" to "All Files" -> Save as coco.names.

After downloading, your project folder should look like this:

/your-project-folder
├── detect_objects.py
├── yolov4-tiny.cfg
├── yolov4-tiny.weights
└── coco.names

Usage
Once the setup is complete, run the main script from your terminal:

python detect_objects.py

A window will open showing your webcam feed with labeled bounding boxes around any detected objects. To stop the program, press the 'q' key.

Acknowledgements
This project uses the YOLO (You Only Look Once) object detection system created by Joseph Redmon.

The YOLOv4 model was developed by Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao.
