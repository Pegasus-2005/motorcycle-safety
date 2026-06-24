import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class LSTMVerificationEngine:
    def __init__(self, model_path: str, train_data_path: str, features: list):
        """
        Initializes the deep learning detector wrapper.
        Applies the safety-critical boundary optimized via F2-score.
        """
        self.features = features
        # Academic Boundary: Configured from the 40-epoch fully converged LSTM run
        self.threshold = 0.9980 
        
        print(f"[DETECTOR] Initializing LSTM Verification Engine with Guard Threshold: {self.threshold}")
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()
        self._bootstrap_scaler_coefficients(train_data_path)

    def _bootstrap_scaler_coefficients(self, path: str):
        """Refits the scaler parameters dynamically to guarantee absolute feature scaling alignment."""
        print("[DETECTOR] Calibrating StandardScaler on training reference baseline features...")
        df = pd.read_csv(path)
        
        # Consistent data engineering transformations
        df['angveldiff'] = df.FW_angvel_Y - df.RW_angvel_Y
        df['angaccdiff'] = df.FW_angacc_Y - df.RW_angacc_Y
        df['sensorLeft'] = np.logical_or(df.SwitchSidesensor_left_road, df.SwitchSidesensor_left_car).astype(int)
        df['sensorRight'] = np.logical_or(df.SwitchSidesensor_right_road, df.SwitchSidesensor_right_car).astype(int)
        df['FW_cnt_Force'] = df.FW_Car_cnt_force + df.FW_Road_cnt_force
        df['RW_cnt_Force'] = df.RW_Car_cnt_force + df.RW_Road_cnt_force
        
        self.scaler.fit(df[self.features])
        print("[DETECTOR] Scaler coefficients successfully cached into memory.")

    def scale_reading(self, raw_reading: np.ndarray) -> np.ndarray:
        """Standardizes live unscaled telemetry vectors before buffer ingestion."""
        # Wrap the raw vector in a temporary DataFrame with explicit feature columns
        # to eliminate scikit-learn feature name warning spam.
        df_temp = pd.DataFrame([raw_reading], columns=self.features)
        return self.scaler.transform(df_temp)[0]

    def verify_state_sequence(self, window_sequence: list) -> tuple:
        """Runs fast millisecond-level inference on the rolling Digital Twin memory state matrix."""
        # Shape required by LSTM layer: (batch_size, window_size, features) -> (1, 50, 14)
        input_tensor = np.array([window_sequence])
        probability = float(self.model.predict(input_tensor, verbose=0)[0][0])
        is_crash = probability >= self.threshold
        return is_crash, probability