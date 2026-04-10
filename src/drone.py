from pymavlink import mavutil
import time

class Drone:
    def __init__(self, port="/dev/ttyUSB0", baud=57600):
        self.mav = mavutil.mavlink_connection(port, baud=baud)
        self.mav.wait_heartbeat()
        print(f"FCU connected — system {self.mav.target_system}")

    def arm_and_takeoff(self, altitude=2.0):
        # Set GUIDED mode (FCU handles altitude hold = anti-gravity)
        self.set_mode("GUIDED")
        time.sleep(1)

        # Arm motors
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        time.sleep(2)

        # Takeoff to target altitude
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        time.sleep(4)  # wait to reach altitude
        print(f"Hovering at {altitude}m — anti-gravity active (GUIDED mode)")

    def set_mode(self, mode_name):
        mode_id = self.mav.mode_mapping()[mode_name]
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

    def send_velocity(self, vx=0, vy=0, vz=0, yaw_rate=0):
        """
        Send body-frame velocity. FCU maintains altitude unless vz != 0.
        vx = forward/back (m/s)
        vy = left/right (m/s)
        vz = up/down (m/s) — keep 0 for altitude hold
        yaw_rate = rotation (rad/s)
        """
        self.mav.mav.set_position_target_local_ned_send(
            int(time.time() * 1000),
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b010111000111,   # use velocity + yaw_rate only
            0, 0, 0,          # position (ignored)
            vx, vy, vz,       # velocity
            0, 0, 0,          # accel (ignored)
            0, yaw_rate       # yaw, yaw_rate
        )

    def hover(self):
        """Stop all movement — drone holds position"""
        self.send_velocity(0, 0, 0, 0)

    def land(self):
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )