
import cv2
import numpy as np
from ultralytics import YOLO

def check_bikes(path):
    print(f"\n--- Checking BIKES in {path} ---")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open.")
        return

    # Skip to frame 100 to avoid static start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        return

    # Use 's' model and lower conf
    model = YOLO("yolov8s.pt")
    results = model(frame, conf=0.3, classes=[2, 3, 5, 7], verbose=False)
    
    count = 0
    bikes = 0
    for r in results:
        count += len(r.boxes)
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if cls == 3:
                bikes += 1
            print(f" - Class: {cls} ({model.names[cls]}), Conf: {conf:.2f}")
    
    print(f"Total detections: {count}")
    print(f"Bikes detected: {bikes}")
    cap.release()

check_bikes("demo1.mp4")
