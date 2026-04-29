# Traffic Monitoring System

A Flask-based traffic monitoring dashboard that runs YOLOv8 object detection with DeepSORT tracking. It supports uploaded videos or live webcam streaming, shows counts, density, speed, risk analysis, and violation logs in a real-time UI.

## Features
- Vehicle detection with YOLOv8
- Multi-object tracking with DeepSORT
- Live dashboard with counts, density, speed, and risk
- Violation logging and stop/reset controls
- Video upload and webcam input

## Tech Stack
- Python 3
- Flask
- OpenCV
- Ultralytics YOLOv8
- deep_sort_realtime

## Project Structure
- app.py - Flask app and streaming pipeline
- detector.py - YOLOv8 detection wrapper
- tracker.py - DeepSORT tracker wrapper
- config.py - Tunable settings and thresholds
- templates/index.html - Frontend dashboard

## Setup
1) Create and activate a virtual environment.
2) Install dependencies.
3) Run the server.

Example:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Usage
- Open http://127.0.0.1:5001 in a browser.
- Upload a video or click Start Camera.
- Use Stop & Reset to clear state and return to the input screen.

## API Endpoints
- GET / - Dashboard UI
- GET /video - MJPEG video stream
- GET /stats - Live stats JSON
- GET /alerts - Violation logs JSON
- POST /upload - Video upload
- GET /webcam - Switch to webcam
- POST /camera_frame - Push webcam frames
- POST /stop - Stop stream and reset state

## Configuration
Edit config.py to tune:
- Model path and allowed classes
- Counting lines and thresholds
- Density thresholds and speed smoothing
- Heatmap and other display options

## Notes
- The default model file yolov8n.pt must exist in the project root.
- For best results, use clear daylight footage with stable camera positioning.

## License
Add a license file if you plan to distribute this project.
