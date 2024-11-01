# Deepfake Detection System for Face-Swap Videos

This project was developed for Smart India Hackathon (SIH) 2024 under the topic: Development of AI/ML-based Solution for Detection of Face-Swap Based Deepfake Videos. It employs advanced AI and machine learning techniques to identify face-swapping manipulations in videos.

![0](https://github.com/user-attachments/assets/b599a992-ae32-4203-97b5-d92bdfc1d49b)


## Table of Contents
Overview
Features
Requirements
Installation
Usage
Model Details
Contributing
License
Overview
With the increasing prevalence of deepfake technology, detecting manipulated media has become critical. This project focuses on developing a reliable, efficient, and scalable AI solution to detect face-swapped deepfake videos by analyzing both visual and audio cues. It provides a confidence score for the detected fakeness, aiming to ensure greater security in digital media.

## Features
Face Detection and Analysis: Analyzes video frames to detect face-swap manipulations.
Audio-Visual Correlation: Incorporates both video and audio analysis for comprehensive detection.
Confidence Score: Displays a confidence score for the detected level of deepfake.
User-Friendly Interface: GUI for video upload and result display.
Requirements
Python 3.x
PyTorch
moviepy
librosa
OpenCV
StreamLit
FastAPI (for backend integration if needed)
For a full list, see requirements.txt.

## Installation
Clone this repository:


git clone https://github.com/yourusername/deepfake-detection
cd deepfake-detection/notebook
Install dependencies:


pip install -r requirements.txt
Set up environment variables in a .env file as needed.

## Usage
Run the main script to start the detection system:


streamlit run .\deepfake_detection.py
Upload a video or provide a URL to analyze.

View the confidence score and analysis results in the GUI.

For more details, refer to the project explanation video.

## Model Details
The project employs transfer learning using state-of-the-art models like T5 and BERT for question answering and document retrieval components. Video frames are processed to detect facial anomalies, while audio cues are examined for inconsistencies.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For significant changes, open an issue first to discuss your ideas.
