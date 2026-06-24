import math
import random
import time


class MotorcycleSimulator:
    

    def __init__(
        self,
        initial_lat: float = 12.9716,
        initial_lon: float = 77.5946,
        initial_speed_kmh: float = 50.0,
    ):
        # Position and motion states
        self.latitude = initial_lat
        self.longitude = initial_lon
        self.speed_mps = initial_speed_kmh / 3.6  # store in m/s
        self.heading = 0.0  # degrees, 0 = North, 90 = East

        # Current simulator profile
        self.current_state = "normal"  # normal, braking, pothole, crash
        self.state_step_counter = 0

        # Constants
        self.GRAVITY = 9.81  # m/s^2

        # Noise levels for normal riding
        self.accel_noise = 0.3
        self.gyro_noise = 1.5

    def set_state(self, state: str) -> None:
        """Manually change the simulator state.

        Supported states: 'normal', 'braking', 'pothole', 'crash'
        """
        if state in ["normal", "braking", "pothole", "crash"]:
            self.current_state = state
            self.state_step_counter = 0

    def step(self) -> dict:
        """Generates the next telemetry data point based on the active state."""
        timestamp = time.time()
        self.state_step_counter += 1

        # Base default parameters
        accel_x = 0.0  # Lateral (left/right)
        accel_y = 0.0  # Longitudinal (forward/back)
        accel_z = self.GRAVITY  # Vertical (up/down)

        gyro_x = 0.0  # Pitch rate (deg/s)
        gyro_y = 0.0  # Roll rate (deg/s)
        gyro_z = 0.0  # Yaw rate (deg/s)

        # Process states
        if self.current_state == "normal":
            # Normal riding updates
            # Occasional small steering adjustments
            yaw_rate = random.uniform(-5.0, 5.0)
            self.heading = (self.heading + yaw_rate * 0.1) % 360.0

            # Slight speed variations
            self.speed_mps += random.uniform(-0.2, 0.2)
            self.speed_mps = max(5.0, min(self.speed_mps, 35.0))  # limit speed

            accel_y = random.uniform(-0.5, 0.5)  # slight accel/decel
            accel_x = (
                self.speed_mps * math.radians(yaw_rate)
            )  # centrifugal accel
            accel_z = self.GRAVITY + random.uniform(-0.2, 0.2)

            gyro_x = random.uniform(-2.0, 2.0)
            gyro_y = random.uniform(-2.0, 2.0)
            gyro_z = yaw_rate

        elif self.current_state == "braking":
            # Sustained deceleration
            deceleration_target = -7.5  # m/s^2 forward deceleration
            self.speed_mps += deceleration_target * 0.1
            if self.speed_mps < 0:
                self.speed_mps = 0.0
                self.set_state("normal")

            accel_y = deceleration_target + random.uniform(-0.5, 0.5)
            # Front fork dives down (pitch rate spike then stabilizes tilted)
            if self.state_step_counter == 1:
                gyro_x = -15.0
            else:
                gyro_x = random.uniform(-2.0, 2.0)

            accel_z = self.GRAVITY + random.uniform(-0.5, 0.5)
            gyro_y = random.uniform(-1.0, 1.0)
            gyro_z = random.uniform(-2.0, 2.0)

        elif self.current_state == "pothole":
            # Sharp, short-lived vertical acceleration spike
            if self.state_step_counter == 1:
                # Compression stage
                accel_z = self.GRAVITY - 18.0
                gyro_x = 25.0
            elif self.state_step_counter == 2:
                # Rebound stage
                accel_z = self.GRAVITY + 22.0
                gyro_x = -20.0
            else:
                # Recovery
                self.set_state("normal")
                accel_z = self.GRAVITY + random.uniform(-1.0, 1.0)

            # Slight deceleration impact
            accel_y = -2.0
            self.speed_mps = max(0.0, self.speed_mps - 0.5 * 0.1)

        elif self.current_state == "crash":
            # Multi-phase crash simulation
            if self.state_step_counter <= 2:
                # Phase 1: High-energy Impact
                accel_x = random.choice([-35.0, 35.0]) + random.uniform(
                    -5.0, 5.0
                )
                accel_y = -40.0 + random.uniform(-5.0, 5.0)
                accel_z = random.uniform(-15.0, 30.0)

                gyro_x = random.uniform(-180.0, 180.0)
                gyro_y = random.uniform(-180.0, 180.0)
                gyro_z = random.uniform(-180.0, 180.0)

                self.speed_mps = max(0.0, self.speed_mps * 0.3)
            elif self.state_step_counter <= 6:
                # Phase 2: Sliding / Tumbling
                accel_x = random.uniform(-10.0, 10.0)
                accel_y = random.uniform(-15.0, 0.0)
                accel_z = random.uniform(-5.0, 15.0)

                gyro_x = random.uniform(-90.0, 90.0)
                gyro_y = random.uniform(-90.0, 90.0)
                gyro_z = random.uniform(-90.0, 90.0)

                self.speed_mps = max(0.0, self.speed_mps - 10.0 * 0.1)
            else:
                # Phase 3: Resting on side (static tilted state)
                self.speed_mps = 0.0
                # When laying flat on its side, the vertical gravity vector is projected to the lateral axis
                accel_x = (
                    9.3  # close to gravity magnitude but pointing sideways
                )
                accel_y = 1.5
                accel_z = 2.0  # minimal z component because bike is flat

                gyro_x = 0.0
                gyro_y = 0.0
                gyro_z = 0.0

        # Inject white noise to IMU signals
        accel_x += random.gauss(0, self.accel_noise)
        accel_y += random.gauss(0, self.accel_noise)
        accel_z += random.gauss(0, self.accel_noise)
        gyro_x += random.gauss(0, self.gyro_noise)
        gyro_y += random.gauss(0, self.gyro_noise)
        gyro_z += random.gauss(0, self.gyro_noise)

        # Update position based on heading and speed (dead reckoning)
        # 1 degree lat ≈ 111,000 meters
        # 1 degree lon ≈ 111,000 * cos(lat) meters
        dt = 0.1  # assuming step runs at ~10Hz
        distance = self.speed_mps * dt
        lat_change = (distance * math.cos(math.radians(self.heading))) / 111000.0
        lon_change = (
            distance * math.sin(math.radians(self.heading))
        ) / (111000.0 * math.cos(math.radians(self.latitude)))

        self.latitude += lat_change
        self.longitude += lon_change

        return {
            "timestamp": timestamp,
            "accel_x": round(accel_x, 3),
            "accel_y": round(accel_y, 3),
            "accel_z": round(accel_z, 3),
            "gyro_x": round(gyro_x, 3),
            "gyro_y": round(gyro_y, 3),
            "gyro_z": round(gyro_z, 3),
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "speed_kmh": round(self.speed_mps * 3.6, 2),
            "state": self.current_state,
        }