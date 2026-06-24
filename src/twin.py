from collections import deque
import math
from typing import Dict, List, Optional


class DigitalTwin:
    

    def __init__(self, max_size: int = 100):
        """Initializes the digital twin with a fixed-size rolling history buffer.

        At 10Hz, a max_size of 100 stores the last 10 seconds of data.
        """
        self.max_size = max_size
        self.history = deque(maxlen=max_size)

        # Virtual sensor and calculated physical states
        self.estimated_roll = 0.0  # degrees
        self.estimated_pitch = 0.0  # degrees
        self.total_acceleration_magnitude = 1.0  # in g

    def update(self, telemetry: dict) -> None:
        """Updates the digital twin state with new telemetry and performs physics-based state estimation."""
        # Extract inputs
        ax = telemetry.get("accel_x", 0.0)
        ay = telemetry.get("accel_y", 0.0)
        az = telemetry.get("accel_z", 9.81)

        # 1. Physics Engine: Calculate Tilt Angles
        # Compute roll (lateral tilt) and pitch (longitudinal tilt) from gravity projection
        # Roll approximation (tilt around longitudinal axis)
        try:
            self.estimated_roll = math.degrees(math.atan2(ax, az))
        except ZeroDivisionError:
            self.estimated_roll = 90.0 if ax > 0 else -90.0

        # Pitch approximation (tilt around lateral axis)
        try:
            denom = math.sqrt(ax**2 + az**2)
            self.estimated_pitch = math.degrees(math.atan2(-ay, denom))
        except ZeroDivisionError:
            self.estimated_pitch = 0.0

        # 2. Physics Engine: Total Acceleration Magnitude
        self.total_acceleration_magnitude = (
            math.sqrt(ax**2 + ay**2 + az**2) / 9.81
        )

        # 3. Create enriched twin state record
        twin_state = telemetry.copy()
        twin_state["estimated_roll"] = round(self.estimated_roll, 2)
        twin_state["estimated_pitch"] = round(self.estimated_pitch, 2)
        twin_state["accel_mag_g"] = round(self.total_acceleration_magnitude, 3)

        # Append to historical buffer
        self.history.append(twin_state)

    def get_latest_state(self) -> Optional[Dict]:
        """Returns the most up-to-date state of the digital twin."""
        if not self.history:
            return None
        return self.history[-1]

    def get_history(self) -> List[Dict]:
        """Returns the entire rolling history window as a list."""
        return list(self.history)

    def get_speed_trend(self) -> float:
        """Calculates the rate of speed change over the buffered period (km/h per second)."""
        if len(self.history) < 2:
            return 0.0

        first = self.history[0]
        last = self.history[-1]

        time_diff = last["timestamp"] - first["timestamp"]
        if time_diff <= 0:
            return 0.0

        speed_diff = last["speed_kmh"] - first["speed_kmh"]
        return speed_diff / time_diff