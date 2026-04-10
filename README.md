# Advanced Tracking — Drone Vision Tracking System

A real-time autonomous drone tracking system that uses computer vision (YOLOv8 + StrongSORT) and MAVLink-based flight control to follow a locked person target.

---

## Project Structure

```
Advanced Tracking/
├── model/               # Place TensorRT / weight files here
│   └── osnet_x0_25_msmt17.pt   (ReID weights — download separately)
└── src/
    ├── main.py          # Entry point — orchestrates all modules
    ├── capture.py       # Thread-safe video capture
    ├── detector.py      # YOLOv8 person detection
    ├── tracker.py       # StrongSORT multi-object tracker with target lock
    ├── pid.py           # PID controllers for yaw, altitude, and forward motion
    └── drone.py         # MAVLink drone interface (arm, takeoff, velocity, land)
```

---

## Source File Reference

### `main.py` — Entry Point
The top-level script that ties all modules together into the main control loop. It should:
1. Instantiate `VideoCapture`, `Detector`, `TargetTracker`, `DroneController`, and `Drone`
2. Arm and take off the drone
3. Continuously grab frames → detect → track → compute PID errors → send velocity commands
4. Land and clean up on exit

> **Note:** This file is currently empty and needs to be implemented.

---

### `capture.py` — Thread-Safe Video Capture

**Class:** `VideoCapture`

Captures frames from a camera device in a background daemon thread so the main loop always gets the freshest frame without blocking.

| Method | Description |
|---|---|
| `__init__(device, width, height, fps)` | Opens the camera at the given device path (default `/dev/video0`) and starts the background reader thread |
| `get_frame()` | Returns the latest frame as a copy (thread-safe), or `None` if no frame yet |
| `stop()` | Stops the background thread and releases the camera |

**Key detail:** Uses `threading.Lock` to prevent race conditions between the reader thread and the main loop.

---

### `detector.py` — Person Detector

**Class:** `Detector`

Wraps a YOLOv8 model (optimised as a TensorRT `.engine` for NVIDIA Orin Nano) to detect people in a frame.

| Method | Description |
|---|---|
| `__init__(model_path)` | Loads the YOLO model (default: `yolov8n.engine`) |
| `detect(frame)` | Runs inference filtering for class `0` (person). Returns an `ndarray` of shape `(N, 6)` with columns `[x1, y1, x2, y2, conf, cls]`, or an empty array if no detections |

---

### `tracker.py` — Target Tracker

**Class:** `TargetTracker`

Wraps **StrongSORT** (from `boxmot`) to track multiple people across frames and lock onto the **first person** detected. Once locked, it follows that track ID exclusively.

| Method | Description |
|---|---|
| `__init__(reid_weights)` | Initialises StrongSORT with ReID weights running on CUDA in FP16 |
| `update(detections, frame)` | Feeds detections into the tracker. Returns `(target_box, lost_count)` where `target_box` is `(x1, y1, x2, y2)` of the locked target or `None` if lost |

**Lock logic:**
- Locks onto the first track ID seen.
- If the target is not visible for **90 consecutive frames (~3 s at 30 fps)**, it resets and re-acquires.

---

### `pid.py` — PID Controllers

#### `PID` — Generic Single-Axis PID

| Parameter | Description |
|---|---|
| `kp`, `ki`, `kd` | Proportional, Integral, Derivative gains |
| `limit` | Anti-windup clamp applied to the integral term |

| Method | Description |
|---|---|
| `compute(error, dt)` | Computes the PID output for a given error and time step `dt` (default 0.033 s) |
| `reset()` | Resets integral and previous error to zero |

#### `DroneController` — Three-Axis Composite Controller

Combines three `PID` instances to produce drone velocity commands from visual error signals.

| Axis | PID | Input | Output |
|---|---|---|---|
| Yaw | `self.yaw` | `error_x` — horizontal pixel offset from frame centre | `yaw_rate` (rad/s) |
| Altitude | `self.alt` | `error_y` — vertical pixel offset from frame centre | `vz` (m/s) |
| Forward | `self.fwd` | `error_area` — bounding box area ratio vs target ratio | `vx` (m/s) |

**Default gains:**

| Controller | Kp | Ki | Kd | Limit |
|---|---|---|---|---|
| Yaw | 0.003 | 0.0 | 0.0005 | 0.5 rad/s |
| Altitude | 0.002 | 0.0 | 0.0003 | 0.5 m/s |
| Forward | 0.6 | 0.0 | 0.05 | 1.0 m/s |

---

### `drone.py` — MAVLink Drone Interface

**Class:** `Drone`

Communicates with the flight controller over serial MAVLink (via `pymavlink`). The drone operates in **GUIDED** mode which provides autonomous altitude hold ("anti-gravity").

| Method | Description |
|---|---|
| `__init__(port, baud)` | Connects to the radio telemetry (default `/dev/ttyUSB0` at 57600 baud) and waits for heartbeat |
| `arm_and_takeoff(altitude)` | Sets GUIDED mode → arms motors → commands takeoff to `altitude` metres (default 2.0 m) |
| `set_mode(mode_name)` | Sends a mode change command to the FCU |
| `send_velocity(vx, vy, vz, yaw_rate)` | Sends body-frame NED velocity setpoints. `vz=0` keeps altitude hold |
| `hover()` | Sends zero velocity on all axes — drone holds current position |
| `land()` | Commands `MAV_CMD_NAV_LAND` to descend and land |

**Velocity convention:**

| Parameter | Axis | Positive direction |
|---|---|---|
| `vx` | Forward / Back | Forward |
| `vy` | Left / Right | Right |
| `vz` | Up / Down | Down (NED convention) |
| `yaw_rate` | Rotation | Clockwise |

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture and frame processing |
| `ultralytics` | YOLOv8 detection |
| `boxmot` | StrongSORT multi-object tracker |
| `torch` | GPU inference and tensor ops |
| `pymavlink` | MAVLink serial communication with FCU |
| `numpy` | Array handling |

Install with:
```bash
pip install opencv-python ultralytics boxmot torch pymavlink numpy
```

> **Hardware note:** The system is optimised for an **NVIDIA Jetson Orin Nano**. TensorRT (`.engine`) model and FP16 inference are used for performance. On a standard PC, replace `yolov8n.engine` with `yolov8n.pt` and set `device="cpu"` or `device="cuda"` in `TargetTracker`.

---

## Hardware & Platform Configuration

### 1. Model & Tracker Performance (Orin Nano vs Local CPU)
The code is currently tuned for CPU testing on Windows. When deploying back to an **NVIDIA Jetson / Orin Nano**:

**In `src/detector.py`**:
Switch the PyTorch weights back to the TensorRT engine for real-time Jetson inference:
```python
# Change from "yolov8n.pt"
model_path="model/fyp_test5_best.engine"
```

**In `src/tracker.py`**:
Change the `StrongSort` initialisation back to use the GPU:
```python
device="cuda",  # Change from "cpu"
half=True       # Change from False to enable FP16 speedup
```

### 2. Camera & Drone Connection (Jetson Linux vs Windows)
**In `src/main.py`**:
- **Camera Device:** Change `device=0` back to Linux format `device="/dev/video0"`
- **Drone Port:** On Linux / Jetson, use `/dev/ttyUSB0` (or similar, like `/dev/ttyACM0`). On Windows, use `COM3`.
- **Drone Enable:** Make sure you actually uncomment `drone = Drone(port="/dev/ttyUSB0")` ! Local testing uses a mock.

Example changes for Jetson:
```python
cam   = VideoCapture(device="/dev/video0") # Jetson Camera
drone = Drone(port="/dev/ttyUSB0")         # Real drone connection
```

---

## Quick Start

1. **Place model weights** in the `model/` folder:
   - `yolov8n.engine` — TensorRT export of YOLOv8n
   - `osnet_x0_25_msmt17.pt` — ReID weights for StrongSORT

2. **Connect the flight controller** via USB serial. Ensure you update the port in `main.py` depending on your OS (e.g., `/dev/ttyUSB0` on Linux, `COM3` on Windows).

3. **Connect the camera** (default `/dev/video0`).

4. **Implement `main.py`** following the orchestration pattern described above, then run:
   ```bash
   python src/main.py
   ```

5. The drone will arm, take off to 2 m, lock onto the first person in view, and follow them using PID-controlled velocity commands.

---

## Control Flow Diagram

```
VideoCapture (thread)
        │
        ▼ frame
    Detector (YOLOv8)
        │
        ▼ detections [x1,y1,x2,y2,conf,cls]
   TargetTracker (StrongSORT)
        │
        ▼ target_box (x1,y1,x2,y2)
  DroneController (3× PID)
   error_x, error_y, error_area
        │
        ▼ vx, vz, yaw_rate
      Drone (MAVLink)
   send_velocity → FCU
```
