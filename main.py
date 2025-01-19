import math
import cv2
import cvzone
from ultralytics import YOLO

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
while True:
    ret, frame = cap.read()
    res = model(frame, stream=True)
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0]*100))/100
            name = int(box.cls[0])
            current_class = classNames[name]

            if current_class == 'car' or current_class == 'motorbike'\
                    or current_class == 'bus' or current_class == 'truck' and conf > 0.25:
                cvzone.putTextRect(frame, f'{conf}{classNames[name]}', (max(20, x1), max(0, y1)), scale=1, thickness=1)
                cvzone.cornerRect(frame, (x1, y1, w, h))
    cv2.imshow('Car', frame)
    cv2.waitKey(0)
