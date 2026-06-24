import math
from typing import Dict, List


class CrashDetector:
    

    def __init__(
        self,
        impact_threshold_g: float = 3.0,
        tilt_threshold_deg: float = 60.0,
        post_impact_window_samples: int = 15,
    ):
        """Initializes the rule-based detector.

        Args:
            impact_threshold_g: Peak g-force required to trigger impact alert.
            tilt_threshold_deg: Lateral tilt angle indicating vehicle laying flat.
            post_impact_window_samples: Number of samples to evaluate after peak to verify crash.
        """
        self.impact_threshold_g = impact_threshold_g
        self.tilt_threshold_deg = tilt_threshold_deg
        self.post_impact_window_samples = post_impact_window_samples

    def analyze(self, history: List[Dict]) -> Dict:
        """Analyzes recent sliding window history to identify and verify crashes,

        filtering out potholes and hard braking events.
        """
        result = {
            "is_crash_detected": False,
            "classification": "normal",
            "confidence": 0.0,
            "reason": "Normal operational parameters",
        }

        if len(history) < 5:
            return result

        latest = history[-1]

        # 1. Check for basic Hard Braking (No high-g impact, but sustained deceleration)
        ay = latest.get("accel_y", 0.0)
        roll = abs(latest.get("estimated_roll", 0.0))
        speed = latest.get("speed_kmh", 0.0)

        # 2. Find Peak Acceleration Magnitude in the window
        peak_idx = -1
        peak_mag = 0.0
        for i, sample in enumerate(history):
            mag = sample.get("accel_mag_g", 1.0)
            if mag > peak_mag:
                peak_mag = mag
                peak_idx = i

        # 3. Handle High-G Impact Events (Anomaly Trigger)
        if peak_mag >= self.impact_threshold_g:
            # We have an impact trigger! Let's check when it occurred.
            samples_since_peak = len(history) - 1 - peak_idx

            if samples_since_peak < 3:
                # Too early to verify, waiting for more data to confirm state
                result["is_crash_detected"] = False
                result["classification"] = "unverified_impact"
                result["confidence"] = 0.5
                result["reason"] = "High G-force impact detected. Verifying..."
                return result

            # Gather data from the post-impact window
            post_impact_samples = history[peak_idx + 1 :]
            avg_roll_post = sum(
                abs(s.get("estimated_roll", 0.0)) for s in post_impact_samples
            ) / len(post_impact_samples)
            final_speed = post_impact_samples[-1].get("speed_kmh", 0.0)

            # Check for Pothole signature
            # Pothole: quick spike, but bike remains upright afterwards, speed is maintained
            is_upright_post = avg_roll_post < 25.0
            is_moving_post = final_speed > 10.0

            if is_upright_post and is_moving_post:
                result["is_crash_detected"] = False
                result["classification"] = "pothole"
                result["confidence"] = 0.85
                result["reason"] = f"Pothole classified. Peak force {peak_mag:.2f}G detected, but vehicle recovered upright."
                return result

            # Check for Verified Crash signature
            # Crash: High-G impact, followed by high tilt (laying flat) and near-zero speed
            is_tilted_flat = avg_roll_post >= self.tilt_threshold_deg
            is_stopped = final_speed < 8.0

            if is_tilted_flat and is_stopped:
                result["is_crash_detected"] = True
                result["classification"] = "crash_verified"
                result["confidence"] = 0.95
                result[
                    "reason"
                ] = f"Crash verified. Impact of {peak_mag:.2f}G, sustained tilt of {avg_roll_post:.1f}°, and vehicle stationary ({final_speed:.1f} km/h)."
                return result

            # If tilted flat but still showing high speed (possibly a sliding event)
            if is_tilted_flat and not is_stopped:
                result["is_crash_detected"] = True
                result["classification"] = "crash_sliding"
                result["confidence"] = 0.85
                result[
                    "reason"
                ] = f"Crash verified (sliding). Impact of {peak_mag:.2f}G, tilt of {avg_roll_post:.1f}°, and vehicle decelerating."
                return result

            # Unclassified transient event
            result["is_crash_detected"] = False
            result["classification"] = "unverified_anomaly"
            result["confidence"] = 0.4
            result[
                "reason"
                ] = f"Transient anomaly. Impact peak {peak_mag:.2f}G, but post-impact profile did not match crash criteria."
            return result

        # 4. Fallback to standard riding states if no high impact was found
        if ay < -6.5 and roll < 20.0:
            result["is_crash_detected"] = False
            result["classification"] = "hard_braking"
            result["confidence"] = 0.90
            result[
                "reason"
            ] = f"Sustained hard braking detected (Deceleration: {ay:.1f} m/s²)."
            return result

        return result