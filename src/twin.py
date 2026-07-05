import collections
import numpy as np

class KinematicDigitalTwin:
    """
    Virtual Entity (VE) Layer: Maintains a continuous, fixed-length 
    memory of the motorcycle's physics to provide temporal context for edge inference.
    """
    def __init__(self, window_size=50, feature_dim=6):
        self.window_size = window_size
        self.feature_dim = feature_dim
        # collections.deque provides O(1) time complexity for appending and popping
        self.buffer = collections.deque(maxlen=window_size)

    def ingest_telemetry(self, feature_vector):
        """Twin Data Fusion Layer: Ingests 6-axis IMU data."""
        if len(feature_vector) != self.feature_dim:
            raise ValueError(f"Dimension mismatch. Expected {self.feature_dim}, got {len(feature_vector)}")
        self.buffer.append(feature_vector)

    def is_synchronized(self):
        """Checks if the twin has enough historical data to form a complete validation sequence."""
        return len(self.buffer) == self.window_size

    def get_state_matrix(self):
        """Exports the synchronized virtual state for the Service Layer."""
        if not self.is_synchronized():
            return None
        return np.array(self.buffer).reshape(1, self.window_size, self.feature_dim)