from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_path="model/fyp_test5_best.engine"):  # TensorRT for Orin Nano
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Returns array of [x1,y1,x2,y2,conf,cls] for persons only (class 2 in custom model)"""
        results = self.model(frame, classes=[2], verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6))
        return results.boxes.data.cpu().numpy()