# detector.py

from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_path, allowed_classes):
        self.model = YOLO(model_path)
        self.allowed_classes = allowed_classes

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()

            cls = int(cls)

            if cls not in self.allowed_classes:
                continue

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class": cls
            })

        return detections