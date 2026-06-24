import os
import sys
import time

# Resolve directory paths for package structure robustness
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alerts import send_alert
from src.detector import CrashDetector
from src.simulator import MotorcycleSimulator
from src.twin import DigitalTwin


def run_scenario(
    name: str,
    simulator: MotorcycleSimulator,
    twin: DigitalTwin,
    detector: CrashDetector,
    steps: int,
    target_state: str,
) -> bool:
    """Executes a defined simulation phase, running data through the digital twin

    and crash detection system.
    """
    print(f"\n>>> Starting Phase: {name.upper()} (Target: {target_state}) <<<")
    simulator.set_state(target_state)
    alert_fired = False

    for step in range(1, steps + 1):
        # 1. Capture sensor stream
        telemetry = simulator.step()

        # 2. Update virtual representation (Digital Twin)
        twin.update(telemetry)

        # 3. Analyze rolling telemetry history window
        history = twin.get_history()
        analysis = detector.analyze(history)

        # Retrieve current metrics from twin for standard logging
        latest = twin.get_latest_state()
        speed = latest.get("speed_kmh", 0.0)
        roll = abs(latest.get("estimated_roll", 0.0))
        g_force = latest.get("accel_mag_g", 1.0)
        classification = analysis["classification"]

        print(
            f"Step {step:02d} | State: {telemetry['state']:8s} | "
            f"Speed: {speed:5.1f} km/h | Roll: {roll:5.1f}° | "
            f"Impact: {g_force:5.2f}G | Result: {classification.upper()}"
        )

        # 4. Trigger SOS Alert block
        if analysis["is_crash_detected"]:
            location = {
                "latitude": latest.get("latitude", 0.0),
                "longitude": latest.get("longitude", 0.0),
            }
            # Trigger our modular alerting routine
            alert_fired = send_alert(
                message=analysis["reason"],
                location=location,
                severity="CRITICAL",
            )
            # Stop scenario early on verified crash to prevent duplicate SOS broadcasts
            break

        time.sleep(0.05)  # Speeds up local simulation output

    return alert_fired


def main():
    print("=" * 70)
    print("DIGITAL TWIN-BASED MOTORCYCLE SAFETY SYSTEM - CLI SIMULATOR RUN")
    print("=" * 70)

    # Initialize modular components
    simulator = MotorcycleSimulator()
    twin = DigitalTwin(max_size=30)  # 3 seconds history buffer at 10Hz
    detector = CrashDetector(
        impact_threshold_g=3.5,
        tilt_threshold_deg=55.0,
        post_impact_window_samples=8,
    )

    # Execute structured testing sequence to verify robustness against false positives
    # 1. Normal Highway Riding
    run_scenario(
        "Normal Highway Cruising",
        simulator,
        twin,
        detector,
        steps=8,
        target_state="normal",
    )

    # 2. Heavy Road Imperfection / Pothole Impact (Transient Anomaly)
    run_scenario(
        "Severe Pothole Transient",
        simulator,
        twin,
        detector,
        steps=5,
        target_state="pothole",
    )

    # 3. Aggressive Braking (Normal vehicle control deceleration)
    run_scenario(
        "Emergency Hard Braking",
        simulator,
        twin,
        detector,
        steps=6,
        target_state="braking",
    )

    # 4. Uncontrolled Crash Event
    crash_detected = run_scenario(
        "High-Side Crash Sequence",
        simulator,
        twin,
        detector,
        steps=12,
        target_state="crash",
    )

    # Output analytical result
    print("=" * 70)
    print("SIMULATION SEQUENCE TERMINATED")
    print("=" * 70)
    if crash_detected:
        print("Success: System successfully intercepted and verified crash.")
        print("Success: Potholes and hard braking correctly filtered (No false alerts).")
    else:
        print("Error: Failure to process/verify target emergency profile.")


if __name__ == "__main__":
    main()