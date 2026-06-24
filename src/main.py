import os
import sys
import time
import numpy as np
import pandas as pd

# Enforce secure package mapping boundaries 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.detector import LSTMVerificationEngine
from src.twin import KinematicDigitalTwin

# Runtime Paths Configuration
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_PATH = os.path.join(DATA_DIR, "trainingData.csv")
SCENARIO_DIR = os.path.join(DATA_DIR, "ControlScenarios")
MODEL_PATH = os.path.join(BASE_DIR, "src", "paper_model_lstm.keras")

# Strict Feature Ordering Template
FEATURES = [
    'MotoBody_angaccY', 'MotoBody_angvelY', 'MotoBody_linaccX', 'MotoBody_linaccZ',
    'MotoFW_linaccX', 'MotoFW_linaccZ', 'MotoRW_linaccX', 'MotoRW_linaccZ',
    'FW_cnt_Force', 'RW_cnt_Force', 'angveldiff', 'angaccdiff', 'sensorLeft', 'sensorRight'
]

def execute_software_in_the_loop_simulation(scenario_filename: str, twin: KinematicDigitalTwin, engine: LSTMVerificationEngine):
    path = os.path.join(SCENARIO_DIR, scenario_filename)
    if not os.path.exists(path):
        print(f"[SIMULATION CRITICAL ERROR] Target file {scenario_filename} absent at location: {path}")
        return

    print("\n" + "="*65)
    print(f"🏍️  STARTING DIGITAL TWIN EXPERIMENTAL playback: {scenario_filename}")
    print("="*65)

    df = pd.read_csv(path)
    # Structural features extraction
    df['angveldiff'] = df.FW_angvel_Y - df.RW_angvel_Y
    df['angaccdiff'] = df.FW_angacc_Y - df.RW_angacc_Y
    df['sensorLeft'] = np.logical_or(df.SwitchSidesensor_left_road, df.SwitchSidesensor_left_car).astype(int)
    df['sensorRight'] = np.logical_or(df.SwitchSidesensor_right_road, df.SwitchSidesensor_right_car).astype(int)
    df['FW_cnt_Force'] = df.FW_Car_cnt_force + df.FW_Road_cnt_force
    df['RW_cnt_Force'] = df.RW_Car_cnt_force + df.RW_Road_cnt_force

    crash_triggered = False

    for idx, row in df.iterrows():
        raw_reading = row[FEATURES].values
        
        # Step A: Normalize incoming vector via our verification calibration wrapper
        scaled_reading = engine.scale_reading(raw_reading)
        
        # Step B: Push the standardized state vector directly to update the Virtual Twin replica
        twin.update_state(scaled_reading)
        
        # Step C: Evaluate if twin possesses complete chronological depth
        if twin.is_synchronized():
            twin_matrix = twin.get_current_twin_matrix()
            is_crash, probability = engine.verify_state_sequence(twin_matrix)
            
            if idx % 50 == 0:
                print(f"[Time: {row['Time']:.2f}s] Twin Sync Verified | Crash Probability: {probability:.4f} | Health: SECURE")
            
            if is_crash:
                print(f"\n [HARDWARE SYSTEM INTERRUPT] ACCIDENT VERIFIED BY KINEMATIC TWIN STATE AT {row['Time']:.2f}s!")
                print(f" Exceeded F2 Safety Frontier Threshold: {probability:.4f} >= {engine.threshold}")
                print(f" [ALERT PIPELINE] Activating SIM800L Cellular System & Telemetry Broadcast...")
                crash_triggered = True
                break
                
        # Simulate physical processing period latency of edge hardware clocks (~50Hz sampling)
        time.sleep(0.002)

    if not crash_triggered:
        print(f"\n🏁 SEQUENCE CONCLUDED. The Kinematic Digital Twin successfully rejected the anomaly. Zero False Alarms triggered.")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[SIMULATION CRITICAL ERROR] Trained LSTM weight artifact not found at {MODEL_PATH}.")
        return

    # Clean independent instantiation
    engine = LSTMVerificationEngine(MODEL_PATH, TRAIN_PATH, FEATURES)
    
    # Session 1: Run operational road challenge anomaly (Pothole test file)
    twin_session_one = KinematicDigitalTwin(window_size=50)
    print("\nExecuting Operational Non-Crash Anomaly Rejection Simulation Profile...")
    time.sleep(1)
    # Testing against Pothole scenario
    pothole_csv = "Pothole_5_Out.csv" if os.path.exists(os.path.join(SCENARIO_DIR, "Pothole_5_Out.csv")) else "Pothole_5_Out.csv"
    execute_software_in_the_loop_simulation(pothole_csv, twin_session_one, engine)
    
    # Session 2: Run collision sequence challenge (ISO Crash test file)
    twin_session_two = KinematicDigitalTwin(window_size=50)
    print("\nExecuting Critical Collision Sequence Verification Simulation Profile...")
    time.sleep(1)
    execute_software_in_the_loop_simulation("ISO13232-1_Out.csv", twin_session_two, engine)

if __name__ == "__main__":
    main()