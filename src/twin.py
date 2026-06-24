from collections import deque

class KinematicDigitalTwin:
    def __init__(self, window_size: int = 50):
        """
        Maintains a fast rolling virtual replica state of vehicle dynamics inside RAM.
        This window tracking provides the temporary historical memory context for the system.
        """
        self.window_size = window_size
        self.rolling_buffer = deque(maxlen=window_size)
        print(f"[DIGITAL TWIN] Active Virtual Kinematic Twin instantiated. Buffer capacity: {window_size} frames.")

    def update_state(self, scaled_reading: list):
        """Ingests a standardized multi-axis sensor data point into the twin memory loop."""
        self.rolling_buffer.append(scaled_reading)

    def is_synchronized(self) -> bool:
        """Blocks verification calls until the window depth has matured completely."""
        return len(self.rolling_buffer) == self.window_size

    def get_current_twin_matrix(self) -> list:
        """Returns the sequential matrix footprint mapping recent riding history."""
        return list(self.rolling_buffer)