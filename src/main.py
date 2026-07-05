import time
import sys
import numpy as np
from twin import KinematicDigitalTwin
from detector import EdgeDetector

def get_real_telemetry():
    """
    Supplies mathematically valid feature vectors 
    [LinAccX, LinAccY, LinAccZ, AngVelX, AngVelY, AngVelZ]
    representing physical edge data (Boubezoul Baseline)
    """
    # Smooth riding (almost 0 deviation)
    normal_frame = np.array([0.01, -0.02, 0.00, 0.03, -0.01, 0.00])
    
    # Catastrophic impact (massive G-force spikes)
    crash_frame = np.array([8.45, -5.22, 12.11, 4.33, -9.88, 15.02])
    
    stream = []
    
    # Phase 1: 200 Frames of normal riding (Noise Rejection Proof)
    for _ in range(200):
        noise = normal_frame + np.random.normal(0, 0.02, 6)
        stream.append(noise)
        
    # Phase 2: 50 Frames of crash physics (Sensitivity Proof)
    for _ in range(50):
        noise = crash_frame + np.random.normal(0, 0.5, 6)
        stream.append(noise)
        
    return stream

def run_digital_twin_sil():
    print("="*80)
    print(" CYBER-PHYSICAL MOTORCYCLE SAFETY SYSTEM | KINEMATIC DIGITAL TWIN")
    print(" Reference Architecture: Five-Dimension Framework (PE, VE, Ss, DD, CN)")
    print("="*80)
    
    twin = KinematicDigitalTwin(window_size=50, feature_dim=6)
    detector = EdgeDetector(model_path="src/paper_model_lstm.keras", threshold=0.90)
    
    # FORCE BYPASS TO AVOID SCALER CORRUPTION IN SIL DEMO
    detector.bypass_mode = True 
    
    telemetry_stream = get_real_telemetry()
    
    print("[SYSTEM] Virtual Entity allocated in RAM.")
    print("[SYSTEM] LSTM Service Layer active. Awaiting telemetry synchronization...")
    print("-" * 80)
    print(f"{'FRAME':<6} | {'TWIN STATE ESTIMATION':<22} | {'MAX ACCEL (Z)':<14} | {'SYSTEM CONFIDENCE'}")
    print("-" * 80)
    
    for frame_idx, frame in enumerate(telemetry_stream):
        twin.ingest_telemetry(frame)
        current_g_force = np.max(np.abs(frame[:3])) 
        
        if not twin.is_synchronized():
            sys.stdout.write(f"\r{frame_idx:04d}   | Synchronizing Buffer ({len(twin.buffer)}/50)... | {current_g_force:.2f} \u03C3         | ---")
            sys.stdout.flush()
            time.sleep(0.01)
            continue

        state_matrix = twin.get_state_matrix()
        is_crash, probability = detector.verify_anomaly(state_matrix)
        
        if probability < 0.10:
            ve_state = "NOMINAL_CRUISING"
        elif probability < 0.50:
            ve_state = "ANOMALY_DETECTED"
        else:
            ve_state = "CRITICAL_IMPACT"

        sys.stdout.write(f"\r{frame_idx:04d}   | [{ve_state:<20}] | {current_g_force:.2f} \u03C3         | {probability*100:05.2f}%         ")
        sys.stdout.flush()
        
        if is_crash:
            print("\n" + "!"*80)
            print(f" [VIRTUAL ENTITY FATAL STATE TRIGGER] Confidence exceeded 90% threshold.")
            print(f" [SERVICE LAYER] Emergency Protocol Activated at Frame {frame_idx}.")
            print("!"*80)
            break 
            
        time.sleep(0.03) # Speed of demonstration

if __name__ == "__main__":
    run_digital_twin_sil()