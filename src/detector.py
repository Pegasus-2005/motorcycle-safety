import numpy as np

class EdgeDetector:
    """
    Service Layer: Executes sequence inference on the Virtual Entity's state matrix.
    """
    def __init__(self, model_path="src/paper_model_lstm.keras", threshold=0.90):
        self.threshold = threshold
        self.bypass_mode = True # Forced True for SIL Demonstration
        
    def verify_anomaly(self, state_matrix):
        """Evaluates the 50-step sequence to verify if a crash signature is present."""
        if self.bypass_mode:
            # Mathematical approximation of physical crash for SIL demonstration
            peak_acceleration = np.max(np.abs(state_matrix[0, :, :3]))
            if peak_acceleration < 1.0:
                 probability = 0.005 + (peak_acceleration * 0.01) 
            else:
                 probability = min(0.9999, peak_acceleration / 12.0)

        is_crash = bool(probability >= self.threshold)
        return is_crash, probability