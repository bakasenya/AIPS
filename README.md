AI Proctoring System (AIPS) is a smart, AI-powered solution designed to maintain academic integrity in online examinations. It leverages computer vision and audio analysis to monitor students during exams, automatically detecting suspicious behaviors and reducing the need for human invigilators.

# 📌 Features
## 🎥 Real-time Video Monitoring
Detects eye movement, face orientation, and the presence of multiple people using webcam footage.

## 🔊 Audio Monitoring
Uses microphone input to detect and flag suspicious sounds or unauthorized speech.

## 🤖 AI & Machine Learning Integration
Employs OpenCV, Dlib, YOLOv3, and Google Speech Recognition API for intelligent detection.

## 📊 Automated Reporting
Generates logs and alerts based on detected anomalies during the examination.

## 💻 Technologies Used
Python

OpenCV

Dlib

YOLOv3 (You Only Look Once)

Google Speech Recognition API

Flask (for web integration)

MySQL (for data storage)

HTML, CSS, JavaScript (frontend components)

🛠 System Architecture
Video Input from webcam is processed via OpenCV and Dlib for facial recognition, eye tracking, and movement analysis.

Audio Input is monitored through the microphone and analyzed using speech recognition techniques.

YOLOv3 is used for object detection — e.g., identifying mobile phones, extra people, or other unauthorized items.

## Flagging Mechanism logs activities such as:

No face detected

Multiple faces detected

Eyes not on screen

Audio disturbances
