import math
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import *


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap = cv2.VideoCapture('./data_set/cars.mp4')
model = YOLO('./yolov8l.pt')
mask = cv2.imread('./mask.png')

# tracking
track = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCount = []
while True:
    ret, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame, mask)
    res = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0]*100))/100
            name = int(box.cls[0])
            current_class = classNames[name]

            if current_class in ['car', 'motorbike', 'bus', 'truck'] and conf > 0.25:
                # cvzone.putTextRect(frame, f'{conf}{classNames[name]}', (max(20, x1), max(0, y1)), scale=1,
                # thickness=1)
                # cvzone.cornerRect(frame, (x1, y1, w, h))
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    results_tracker = track.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    for res in results_tracker:
        x1, y1, x2, y2, id = map(int, res)
        w, h = x2-x1, y2-y1
        print(res)
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=5, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'{int(id)}', (max(20, x1), max(0, y1)), scale=1, thickness=1)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx,cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-30 < cy < limits[1]+30:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        cvzone.putTextRect(frame, f' Count: {len(totalCount)}', (50, 50))

    cv2.imshow('Car', frame)
    cv2.waitKey(0)
