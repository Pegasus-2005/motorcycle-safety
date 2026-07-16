import collections
import numpy as np 
class KinematicDigitalTwin:

    def __init__(self, window_size=50, feature_dim=6):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = collections.deque(maxlen=window_size)

    def ingest_telemetry(self, feature_vector):
        if len(feature_vector) != self.feature_dim:
            raise ValueError(f"Dimension mismatch. Expected {self.feature_dim}, got {len(feature_vector)}")
        self.buffer.append(feature_vector)

    def is_synchronized(self):
        return len(self.buffer) == self.window_size

    def get_state_matrix(self):
        if not self.is_synchronized():
            return None
        return np.array(self.buffer).reshape(1, self.window_size, self.feature_dim)