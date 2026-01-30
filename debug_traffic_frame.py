
import cv2
import numpy as np
from ultralytics import YOLO

def check_video_frame(path, frame_idx):
    print(f"\n--- Checking {path} at frame {frame_idx} ---")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        return

    print(f"Mean pixel value: {np.mean(frame):.2f}")
    
    model = YOLO("yolov8n.pt")
    results = model(frame, conf=0.1, verbose=False) # very low conf
    
    count = 0
    for r in results:
        count += len(r.boxes)
        for box in r.boxes:
             print(f" - Class: {int(box.cls)} ({model.names[int(box.cls)]}), Conf: {float(box.conf):.2f}")
    
    print(f"Total detections: {count}")
    cap.release()

check_video_frame("traffic.mp4", 500)
