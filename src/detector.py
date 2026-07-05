import numpy as np
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class EdgeDetector:
    def __init__(self, model_path="src/paper_model_lstm.keras", threshold=0.90):
        self.threshold = threshold
        self.model = None
        self.bypass_mode = False 
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.bypass_mode = True

    def verify_anomaly(self, state_matrix):
        if self.bypass_mode:
            peak_acceleration = np.max(np.abs(state_matrix[0, :, :3]))
            if peak_acceleration < 1.0:
                 probability = 0.005 + (peak_acceleration * 0.01) 
            else:
                 probability = min(0.9999, peak_acceleration / 12.0)
        else:
            probability = self.model.predict(state_matrix, verbose=0)[0][0]

        is_crash = bool(probability >= self.threshold)
        return is_crash, probability