class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit    = limit
        self.integral = 0.0
        self.prev_err = 0.0

    def compute(self, error, dt=0.033):
        self.integral += error * dt
        # Anti-windup clamp
        self.integral = max(-self.limit, min(self.limit, self.integral))
        derivative    = (error - self.prev_err) / max(dt, 1e-6)
        self.prev_err = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0


class DroneController:
    """Three PIDs: yaw (left/right), altitude (up/down), forward (distance)"""
    def __init__(self):
        self.yaw = PID(kp=0.003, ki=0.0,    kd=0.0005, limit=0.5)   # rad/s
        self.alt = PID(kp=0.002, ki=0.0,    kd=0.0003, limit=0.5)   # m/s vz
        self.fwd = PID(kp=0.6,   ki=0.0,    kd=0.05,   limit=1.0)   # m/s vx

    def compute(self, error_x, error_y, error_area, dt=0.033):
        """
        error_x    = target cx - frame center x  (pixels)
        error_y    = target cy - frame center y  (pixels)
        error_area = bbox_area_ratio - target_ratio (e.g. 0.08)
        Returns (vx, vz, yaw_rate)
        """
        yaw_rate = self.yaw.compute(error_x,   dt)
        vz       = self.alt.compute(error_y,   dt)   # positive = move down
        vx       = self.fwd.compute(-error_area, dt)  # negative = move closer
        return vx, vz, yaw_rate

    def reset(self):
        self.yaw.reset()
        self.alt.reset()
        self.fwd.reset()