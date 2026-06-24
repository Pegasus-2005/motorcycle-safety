import os
import sys
import time
import pandas as pd
import streamlit as st

# Ensure the parent directory is in the path to allow clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alerts import send_alert
from src.detector import CrashDetector
from src.simulator import MotorcycleSimulator
from src.twin import DigitalTwin

# Page configuration
st.set_page_config(
    page_title="Digital Twin Motorcycle Safety Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State Variables
if "simulator" not in st.session_state:
    st.session_state.simulator = MotorcycleSimulator()
if "twin" not in st.session_state:
    st.session_state.twin = DigitalTwin(max_size=50)
if "detector" not in st.session_state:
    st.session_state.detector = CrashDetector()
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False
if "system_logs" not in st.session_state:
    st.session_state.system_logs = []


def add_log(classification: str, reason: str):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    log_entry = f"[{timestamp}] State: {classification.upper()} - {reason}"
    if not st.session_state.system_logs or st.session_state.system_logs[0] != log_entry:
        st.session_state.system_logs.insert(0, log_entry)


# Sidebar controls
st.sidebar.title("🎮 Simulation Controller")

st.sidebar.subheader("Select Scenario")
scenario = st.sidebar.selectbox(
    "Active Scenario Mode",
    ["normal", "braking", "pothole", "crash"],
    index=0,
    help="Changes the physical profile of the simulated motorcycle.",
)

# Update state if scenario changed
if scenario != st.session_state.simulator.current_state:
    st.session_state.simulator.set_state(scenario)
    add_log("system", f"Scenario manual transition to '{scenario}'")

st.sidebar.divider()

# Simulation Controls
st.sidebar.subheader("Execution Controls")
run_auto = st.sidebar.checkbox("Start Live Stream (10 Hz)", value=False)
step_button = st.sidebar.button("Step Frame ➡️")

st.sidebar.divider()

# Reset Button
if st.sidebar.button("Reset Digital Twin"):
    st.session_state.simulator = MotorcycleSimulator()
    st.session_state.twin = DigitalTwin(max_size=50)
    st.session_state.alert_triggered = False
    st.session_state.system_logs = []
    st.sidebar.success("Twin and simulator state reset.")
    st.rerun()

# Execute physics step
telemetry = None
if step_button or run_auto:
    # 1. Retrieve simulated hardware sensors
    telemetry = st.session_state.simulator.step()

    # 2. Update Digital Twin (virtual model processing)
    st.session_state.twin.update(telemetry)

    # 3. Analyze updated twin with rule detector
    twin_history = st.session_state.twin.get_history()
    analysis_result = st.session_state.detector.analyze(twin_history)

    # Get latest enriched state from twin
    latest_twin_state = st.session_state.twin.get_latest_state()

    # Log changes
    add_log(analysis_result["classification"], analysis_result["reason"])

    # 4. Trigger alert system if verified crash is found and not yet triggered
    if (
        analysis_result["is_crash_detected"]
        and not st.session_state.alert_triggered
    ):
        st.session_state.alert_triggered = True
        location_data = {
            "latitude": latest_twin_state.get("latitude", 0.0),
            "longitude": latest_twin_state.get("longitude", 0.0),
        }
        send_alert(
            message=analysis_result["reason"],
            location=location_data,
            severity="CRITICAL",
        )

# Main Dashboard Interface
st.title("🏍️ Motorcycle Digital Twin Safety Dashboard")
st.markdown(
    "**Academic BTech Project Prototype** — Real-time crash verification using temporal multi-axis analysis."
)

latest_state = st.session_state.twin.get_latest_state()

if latest_state is None:
    st.info("Click 'Step Frame' or start 'Live Stream' in the sidebar to generate data.")
else:
    # Key Status Card Displays
    m1, m2, m3, m4 = st.columns(4)

    # Resolve crash status text & color code
    is_crashed = st.session_state.alert_triggered
    status_label = "💥 CRASHED" if is_crashed else "🟢 SECURE"

    m1.metric("System Safety Status", status_label)
    m2.metric("Vehicle Speed", f"{latest_state['speed_kmh']:.1f} km/h")
    m3.metric("Lean Angle (Roll)", f"{latest_state['estimated_roll']:.1f}°")
    m4.metric("G-Force Vector", f"{latest_state['accel_mag_g']:.2f} G")

    st.divider()

    # Dual Column layouts for Charts and Live Models
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📊 Temporal Physics Trends")

        # Convert twin's rolling history buffer into a pandas dataframe
        hist_df = pd.DataFrame(st.session_state.twin.get_history())

        if not hist_df.empty:
            # Drop unnecessary columns for plotting
            plot_df = hist_df[
                [
                    "accel_x",
                    "accel_y",
                    "accel_z",
                    "estimated_roll",
                    "speed_kmh",
                    "timestamp",
                ]
            ].copy()
            plot_df["time_sec"] = plot_df["timestamp"] - plot_df["timestamp"].iloc[0]

            # Linear Acceleration Chart
            st.markdown("**Tri-Axial Linear Acceleration (m/s²)**")
            st.line_chart(plot_df.set_index("time_sec")[["accel_x", "accel_y", "accel_z"]])

            # Angular Orientation Chart
            st.markdown("**Estimated Tilt / Yaw Stability Index (Degrees)**")
            st.line_chart(plot_df.set_index("time_sec")[["estimated_roll"]])
        else:
            st.caption("Insufficient historical data for charting.")

    with c2:
        st.subheader("🧭 Digital Twin State")

        # Layout structured indicators
        st.markdown(f"**GPS Coordinates:** `{latest_state['latitude']}, {latest_state['longitude']}`")
        st.markdown(f"**Current Scenario Class:** `{latest_state['state'].upper()}`")

        # Visual indicator bar representing tilt
        roll_val = abs(latest_state["estimated_roll"])
        st.progress(min(1.0, roll_val / 90.0), text=f"Lean Angle Severity: {roll_val:.1f}°")

        st.subheader("📜 Live Diagnostics Event Log")
        log_box = "\n".join(st.session_state.system_logs[:10])
        st.text_area(
            label="System classification outcomes over time:",
            value=log_box if log_box else "Awaiting live metrics...",
            height=200,
            disabled=True,
        )

    st.divider()

    # Raw telemetry display section
    st.subheader("📋 Rolling History Frame Buffer")
    df_view = hist_df.copy()
    if not df_view.empty:
        # Reorder columns for readability
        ordered_cols = [
            "timestamp",
            "state",
            "speed_kmh",
            "accel_mag_g",
            "estimated_roll",
            "estimated_pitch",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "latitude",
            "longitude",
        ]
        df_view = df_view[[c for c in ordered_cols if c in df_view.columns]]
        st.dataframe(df_view.tail(15), use_container_width=True)

# Loop update logic for auto-run mode in Streamlit
if run_auto:
    time.sleep(0.1)  # Target ~10Hz refresh rate
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()