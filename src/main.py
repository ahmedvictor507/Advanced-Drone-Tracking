import time
import cv2
from capture  import VideoCapture
from detector import Detector
from tracker  import TargetTracker
from pid      import DroneController
from drone    import Drone

TARGET_AREA_RATIO = 0.08   # ideal: target fills 8% of frame
SCAN_YAW_RATE     = 0.25   # rad/s slow scan when lost

class MockDrone:
    def arm_and_takeoff(self, altitude):
        pass
    def send_velocity(self, vx=0, vy=0, vz=0, yaw_rate=0):
        pass
    def hover(self):
        pass
    def land(self):
        pass

click_coords = None

def mouse_callback(event, x, y, flags, param):
    global click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coords = (x, y)

def main():
    # --- Init ---
    cam        = VideoCapture(device=0) # Use 0 for default webcam testing
    detector   = Detector(model_path="model/fyp_test5_best.pt") # Use generic YOLOv8n.pt for CPU testing
    tracker    = TargetTracker()
    controller = DroneController()
    
    # Use MockDrone for Windows local testing
    # Switch to `Drone(port="/dev/ttyUSB0")` on Linux
    drone = MockDrone()
    # drone = Drone(port="/dev/ttyUSB0")
    
    drone.arm_and_takeoff(altitude=2.0)

    print("Starting tracking loop... (Drone connection is currently disabled for local testing)")
    print("Tracking started — press Q to land")
    
    cv2.namedWindow("Drone Tracker")
    cv2.setMouseCallback("Drone Tracker", mouse_callback)

    prev_time = time.time()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            now = time.time()
            dt  = now - prev_time
            prev_time = now

            h, w = frame.shape[:2]

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track — get all active tracks
            tracks = tracker.update(detections, frame)

            # 2.5 Handle manual selection via mouse click
            global click_coords
            if click_coords is not None:
                cx_click, cy_click = click_coords
                click_coords = None
                for t in tracks:
                    tx1, ty1, tx2, ty2, tid = t
                    if tx1 <= cx_click <= tx2 and ty1 <= cy_click <= ty2:
                        tracker.locked_id = tid
                        tracker.lost_count = 0
                        print(f"Manually locked onto Target ID: {tid}")
                        break

            # 3. Control & Draw
            target_box = None
            for t in tracks:
                x1, y1, x2, y2, tid = t
                if tid == tracker.locked_id:
                    target_box = (x1, y1, x2, y2)
                    
                    # Compute PID and Control
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    box_area_ratio = (x2-x1) * (y2-y1) / (w * h)

                    error_x    = cx - w / 2
                    error_y    = cy - h / 2
                    error_area = box_area_ratio - TARGET_AREA_RATIO

                    vx, vz, yaw_rate = controller.compute(error_x, error_y, error_area, dt)
                    drone.send_velocity(vx=vx, vy=0, vz=vz, yaw_rate=yaw_rate)

                    # Draw actively tracked target (Green)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 6, (0,0,255), -1)
                    cv2.putText(frame, f"LOCKED ID:{tid}", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    # Draw unselected targets (Gray)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (150,150,150), 1)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

            if target_box is None:
                if tracker.locked_id is not None and tracker.lost_count < 90:
                    # Slowly rotate to scan for target
                    controller.reset()
                    drone.send_velocity(yaw_rate=SCAN_YAW_RATE)
                    cv2.putText(frame, "SCANNING...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                else:
                    # Fully lost or no target selected — hover in place
                    drone.hover()
                    cv2.putText(frame, "WAITING / HOVERING", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Drone Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Landing...")
        drone.land()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()