from boxmot import StrongSort
import numpy as np
import torch

class TargetTracker:
    def __init__(self, reid_weights="osnet_x0_25_msmt17.pt"):
        self.tracker = StrongSort(
            reid_weights=reid_weights,
            device="cpu", #Change to CUDA on Jetson
            half=False         # Disabled FP16 since we use CPU
        )
        self.locked_id  = None
        self.lost_count = 0
        self.MAX_LOST   = 90  # ~3 seconds at 30fps before giving up

    def update(self, detections, frame):
        """
        Returns all active tracks: list of [x1, y1, x2, y2, track_id]
        And manages lost_count for locked_id.
        """
        if len(detections):
            tracks = self.tracker.update(detections, frame)
        else:
            tracks = []

        parsed_tracks = []
        for t in tracks:
            # Parse track data
            x1,y1,x2,y2,tid = int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4])
            parsed_tracks.append([x1, y1, x2, y2, tid])

        # Manage lost count for the locked ID
        if self.locked_id is not None:
            found = False
            for t in parsed_tracks:
                if t[4] == self.locked_id:
                    found = True
                    self.lost_count = 0
                    break
            
            if not found:
                self.lost_count += 1
                
            # Give up lock after MAX_LOST frames — reacquire
            if self.lost_count > self.MAX_LOST:
                print(f"Target {self.locked_id} lost — ready for new selection")
                self.locked_id  = None
                self.lost_count = 0

        return parsed_tracks