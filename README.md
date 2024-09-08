# CogniAble_NLP-project

Person Detector and Tracker Inference Script
Overview
This project involves developing a person detection and tracking system specifically designed to monitor children and adults in a video. The main objectives are to:

Detect and assign unique IDs to individuals (children and adults).
Track these individuals throughout the video, even in cases of occlusion or re-entry.
Identify different children with Autism Spectrum Disorder (ASD) and therapists.
Analyze behaviors, emotions, and engagement levels over time.
The system utilizes the YOLO (You Only Look Once) object detection algorithm in conjunction with OpenCV for real-time processing and tracking.

Features
Real-time person detection and tracking
Unique ID assignment to individuals
Robust handling of re-entry and occlusion
Behavior and emotion analysis
Engagement level monitoring
Visualization tools for tracking and analysis
Requirements
The following software and libraries are required to run the project:

Python 3.x
OpenCV
YOLOv5 (or any other YOLO version you prefer)
NumPy
Pandas
SciPy
Matplotlib (for visualization)
PyTorch (for YOLO model)
Scikit-learn (optional, for further analysis)
