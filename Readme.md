# **Car Counter Using OpenCV**

  This project is a car counter application developed using OpenCV and YOLO (You Only Look Once) object detection. It processes video frames to detect and count vehicles crossing a designated line, providing accurate vehicle tracking and statistics.

# Features
Real-Time Object Detection: Leverages YOLO to detect vehicles such as cars, motorbikes, buses, and trucks.

Vehicle Tracking: Tracks objects using SORT (Simple Online and Realtime Tracking) for robust multi-object tracking.

Customizable Detection Zones: Users can define specific zones for vehicle counting using masks.

Graphical Overlay: Adds graphics and statistics to the video feed for better visualization.

Efficient Performance: Processes frames at a reduced resolution for detection.

# Requirements

Python 3.8+

OpenCV

Ultralytics YOLO

cvzone

SORT library

# Installation

Clone the repository:

git clone <repository_url>

cd car-counter

# Install the required dependencies:

pip install -r requirements.txt

Download the YOLO model weights (e.g., yolov8l.pt) and place them in the project directory.

# Usage
Place the video file (e.g., traffic_video.mp4) in the project folder.

Run the main script:  
python main.py

Press the q key to stop the video during playback.

Files in the Repository
main.py: The main script that performs vehicle counting.

mask.png: A binary mask image for defining the region of interest.

graphics.png: A graphic overlay added to the video(not included).

traffic_video.mp4: Example video file for testing.

yolov8l.pt: YOLOv8 model weights (not included, must be downloaded separately).

# Output
The program displays the video with detected vehicles, tracking IDs, and a count of vehicles crossing the defined line.

# Key Code Functionality
## Video Reading: 
cap = cv2.VideoCapture('traffic_video.mp4')

## Object Detection: 
res = model(imgRegion, stream=True)

## Object Tracking: 
results_tracker = track.update(detections)

## Stopping Video: 
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Customization

Change the Detection Line: Modify the limits variable in main.py.

Adjust Detection Classes: Edit the if current_class in [...] condition to include/exclude specific objects.

# Future Improvements
Add support for multiple detection zones.

Integrate with a database for storing vehicle counts.

Improve performance for high-resolution videos.

# License
This project is open-source and available under the .

For any queries, feel free to reach out through GitHub Issues.

