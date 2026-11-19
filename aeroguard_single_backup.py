import os
import csv
import time
import msvcrt
from collections import deque
from datetime import datetime, timezone

import joblib
import pandas as pd
from opensky_api import OpenSkyApi, TokenManager
from transmitter import activate_transmitter


# ============================================================
# AEROGUARD: SINGLE AIRCRAFT MONITOR
# ============================================================

MODEL_FILE = "aeroguard_model_10.pkl"

COLLECTION_INTERVAL = 10
RECORDING_MINUTES = 45
RECORDING_SECONDS = RECORDING_MINUTES * 60
RECORDING_SIZE = 270

AI_WINDOW = 10
AI_HISTORY_SIZE = 30

ANOMALY_REQUIRED = 5
EMERGENCY_CONFIDENCE = 85.0

os.makedirs("data/emergencies", exist_ok=True)
os.makedirs("data/ground_station", exist_ok=True)

print("\n" + "=" * 60)
print("             AEROGUARD")
print("       SINGLE AIRCRAFT MONITOR")
print("=" * 60)

print("\nLoading AI model...")
model = joblib.load(MODEL_FILE)

print("Connecting to OpenSky using credentials...")
token_manager = TokenManager.from_json_file("credentials.json")
api = OpenSkyApi(token_manager=token_manager)

print("AI model loaded successfully.")
print("Authenticated OpenSky connection configured.")


# ============================================================
# AIRCRAFT SELECTION
# ============================================================

def find_aircraft():
    try:
        states = api.get_states()

        if states is None or not states.states:
            return []

        aircraft = []

        for state in states.states:
            if (
                state.icao24
                and state.latitude is not None
                and state.longitude is not None
                and state.geo_altitude is not None
                and state.velocity is not None
                and state.true_track is not None
                and state.vertical_rate is not None
                and state.on_ground is False
            ):
                aircraft.append(state)

        return aircraft[:10]

    except Exception as exc:
        print(f"OpenSky search error: {exc}")
        return []


print("\nSearching for aircraft...")

aircraft_list = []

while not aircraft_list:
    aircraft_list = find_aircraft()

    if not aircraft_list:
        print("No aircraft found. Retrying in 10 seconds...")
        time.sleep(10)

print("\nAvailable aircraft:\n")

for i, aircraft in enumerate(aircraft_list, start=1):
    callsign = (aircraft.callsign or "").strip() or "UNKNOWN"

    print(
        f"{i}. {aircraft.icao24.lower()} {callsign} "
        f"ALT {aircraft.geo_altitude:.0f} m "
        f"SPEED {aircraft.velocity:.1f} m/s"
    )

while True:
    try:
        choice = int(
            input(f"\nSelect aircraft (1-{len(aircraft_list)}): ")
        )

        if 1 <= choice <= len(aircraft_list):
            break

        print("Invalid selection.")

    except ValueError:
        print("Enter a number.")

selected = aircraft_list[choice - 1]
AIRCRAFT_ID = selected.icao24.lower()
CALLSIGN = (selected.callsign or "").strip() or "UNKNOWN"

print("\n" + "=" * 60)
print("SELECTED AIRCRAFT")
print("=" * 60)
print("ICAO24 :", AIRCRAFT_ID)
print("Callsign:", CALLSIGN)
print("=" * 60)


# ============================================================
# RECORDING AND AI STATE
# ============================================================

recording = deque()
ai_history = deque(maxlen=AI_HISTORY_SIZE)

last_observation_timestamp = None
last_ai_window_end = None

anomaly_count = 0
emergency_triggered = False
test_emergency = False


# ============================================================
# KEYBOARD COMMAND
# ============================================================

def check_test_command():
    global test_emergency, emergency_triggered

    while msvcrt.kbhit():
        key = msvcrt.getwch()

        # Handle special keys
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue

        # Accept E only once, and ignore it after a trigger
        if key.lower() == "e":
            if emergency_triggered or test_emergency:
                continue

            test_emergency = True
            print("\nTEST EMERGENCY COMMAND RECEIVED.")


# ============================================================
# SAVE RECORDING
# ============================================================

def save_previous_recording():
    """Save the current rolling buffer and return its path on success."""

    if not recording:
        print("ERROR: No recording available to save.")
        return None

    folder = os.path.join("data", "emergencies", AIRCRAFT_ID)
    filename = os.path.join(folder, "previous_45_minutes.csv")

    try:
        os.makedirs(folder, exist_ok=True)

        df = pd.DataFrame(list(recording))
        df = df.drop(columns=["timestamp_unix"], errors="ignore")

        if df.empty:
            print("ERROR: Recording is empty.")
            return None

        df.to_csv(filename, index=False)

        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            print("ERROR: Recording file is missing or empty after saving.")
            return None

        print(f"\nRecording saved: {filename}")
        print(f"Recorded observations: {len(recording)}")

        if len(recording) < RECORDING_SIZE:
            print(
                "WARNING: Partial recording. Fewer than 270 observations "
                "are available; a complete 45-minute history is not guaranteed."
            )

        # Calculate actual coverage from timestamps.
        timestamps = pd.to_numeric(
            pd.Series([item.get("timestamp_unix") for item in recording]),
            errors="coerce",
        ).dropna()

        if len(timestamps) >= 2:
            coverage_minutes = (
                timestamps.max() - timestamps.min()
            ) / 60.0

            print(
                f"Actual recording time span: "
                f"{coverage_minutes:.1f} minutes"
            )

        return filename

    except (OSError, ValueError, TypeError) as error:
        print(f"ERROR: Recording save failed: {error}")
        return None


# ============================================================
# POST-TRIGGER TELEMETRY
# ============================================================

def append_post_trigger_telemetry(record):
    """Append new observations to the simulated ground station after triggering."""

    if not emergency_triggered:
        return

    folder = os.path.join("data", "ground_station", AIRCRAFT_ID)
    filename = os.path.join(folder, "post_trigger_telemetry.csv")

    try:
        os.makedirs(folder, exist_ok=True)

        needs_header = (
            not os.path.isfile(filename)
            or os.path.getsize(filename) == 0
        )

        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(record.keys())
            )

            if needs_header:
                writer.writeheader()

            writer.writerow(record)

    except (OSError, ValueError, TypeError) as error:
        print(f"Post-trigger telemetry logging failed: {error}")


# ============================================================
# TRANSMIT TEST EMERGENCY
# ============================================================

def trigger_test_emergency():
    global test_emergency, emergency_triggered

    if not test_emergency or emergency_triggered:
        return

    test_emergency = False
    emergency_triggered = True

    print("\n" + "=" * 60)
    print("TEST EMERGENCY ACTIVATED")
    print("=" * 60)

    recording_file = save_previous_recording()
    issues = ["TEST EMERGENCY"]

    if recording_file is None:
        issues.append(
            "Recording save failed or no observations were available; "
            "the emergency report will not contain the recording."
        )

    transmission_ok = activate_transmitter(
        aircraft_id=AIRCRAFT_ID,
        callsign=CALLSIGN,
        issues=issues,
        recording=recording_file,
    )

    print("TEST EMERGENCY COMPLETE.")
    print(
        f"Transmitter result: "
        f"{'SUCCESS' if transmission_ok else 'FAILED'}"
    )
    print("This was a simulation, not a real ATC transmission.")


# ============================================================
# AI FEATURE CREATION
# ============================================================

def create_features(history):
    df = pd.DataFrame(history)

    # Match the population standard deviation used in training.
    features = {
        "altitude_mean": df["baro_altitude"].mean(),
        "altitude_std": df["baro_altitude"].std(ddof=0),

        "altitude_change": (
            df["baro_altitude"].iloc[-1]
            - df["baro_altitude"].iloc[0]
        ),

        "velocity_mean": df["velocity"].mean(),
        "velocity_std": df["velocity"].std(ddof=0),

        "velocity_change": (
            df["velocity"].iloc[-1]
            - df["velocity"].iloc[0]
        ),

        "heading_mean": df["heading"].mean(),
        "heading_std": df["heading"].std(ddof=0),

        "vertical_rate_mean": df["vertical_rate"].mean(),
        "vertical_rate_std": df["vertical_rate"].std(ddof=0),
        "vertical_rate_min": df["vertical_rate"].min(),
        "vertical_rate_max": df["vertical_rate"].max(),
    }

    return pd.DataFrame([features])


# ============================================================
# AI OBSERVATION WINDOW
# ============================================================

def get_ai_window():
    if len(ai_history) < AI_WINDOW:
        print(
            f"AI DEBUG: Have {len(ai_history)}/"
            f"{AI_WINDOW} fresh observations."
        )
        return []

    # Use the latest 10 distinct observations.
    # Actual elapsed time is measured, not assumed.
    recent = list(ai_history)[-AI_WINDOW:]

    timestamps = [
        item["timestamp_unix"]
        for item in recent
    ]

    span = timestamps[-1] - timestamps[0]

    print(
        f"AI DEBUG: {len(recent)} distinct observations; "
        f"actual time span={span:.1f} seconds."
    )

    return recent


# ============================================================
# AI PREDICTION AND EMERGENCY HANDLING
# ============================================================

def run_ai():
    global anomaly_count
    global emergency_triggered
    global last_ai_window_end

    window = get_ai_window()

    if len(window) < AI_WINDOW:
        return

    window_end = window[-1]["timestamp_unix"]

    # Do not evaluate the same window endpoint repeatedly.
    if last_ai_window_end == window_end:
        return

    last_ai_window_end = window_end

    features = create_features(window)

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    emergency_index = (
        classes.index(1) if 1 in classes else None
    )

    emergency_probability = (
        float(probabilities[emergency_index]) * 100
        if emergency_index is not None
        else 0.0
    )

    data_span = (
        window[-1]["timestamp_unix"]
        - window[0]["timestamp_unix"]
    )

    print("\nAEROGUARD AI RESULT")
    print(
        "AI Prediction:",
        "ANOMALY" if prediction == 1 else "NORMAL"
    )
    print(
        f"Emergency probability: "
        f"{emergency_probability:.2f}%"
    )
    print(f"AI data span: {data_span:.0f} seconds")

    # Both conditions must be satisfied for every qualifying window.
    if (
        prediction != 1
        or emergency_probability < EMERGENCY_CONFIDENCE
    ):
        anomaly_count = 0
        print("Emergency streak reset: criteria not met.")
        return

    anomaly_count += 1

    print(
        f"Qualified emergency streak: "
        f"{anomaly_count}/{ANOMALY_REQUIRED}"
    )

    if (
        anomaly_count >= ANOMALY_REQUIRED
        and not emergency_triggered
    ):
        emergency_triggered = True

        print("\n" + "=" * 60)
        print("!!! AI EMERGENCY CONDITION TRIGGERED !!!")
        print("=" * 60)

        recording_file = save_previous_recording()
        issues = ["AI detected abnormal flight behaviour"]

        if recording_file is None:
            issues.append(
                "Recording save failed or no observations were available; "
                "the emergency report will not contain the recording."
            )

        transmission_ok = activate_transmitter(
            aircraft_id=AIRCRAFT_ID,
            callsign=CALLSIGN,
            issues=issues,
            recording=recording_file,
        )

        print(
            "Emergency report operation: "
            f"{'SUCCESS' if transmission_ok else 'FAILED'}"
        )
        print(
            "This is a research prototype with simulated transmission, "
            "not a real ATC connection."
        )


# ============================================================
# MAIN MONITORING LOOP
# ============================================================

print("\nStarting AeroGuard monitoring...")
print("Aircraft:", AIRCRAFT_ID)
print("Callsign:", CALLSIGN)
print("Recording window: 45 minutes")
print(
    f"Recording retention: up to "
    f"{RECORDING_SECONDS} seconds"
)
print(
    f"Collection polling interval: "
    f"{COLLECTION_INTERVAL} seconds"
)
print(
    f"AI window: latest {AI_WINDOW} observations; "
    "elapsed time varies"
)
print(
    f"Emergency rule: {ANOMALY_REQUIRED} consecutive "
    "qualifying windows"
)
print(
    f"Emergency confidence threshold: "
    f"{EMERGENCY_CONFIDENCE:.1f}%"
)
print("Press E (no Enter needed) to trigger TEST EMERGENCY.")
print("Press Ctrl+C to stop.")
print(
    "Research prototype only. No real ATC connection.\n"
)
print("=" * 60)

next_poll = time.monotonic()

while True:
    try:
        check_test_command()
        trigger_test_emergency()

        now = time.monotonic()

        if now < next_poll:
            time.sleep(0.1)
            continue

        next_poll = time.monotonic() + COLLECTION_INTERVAL

        states = api.get_states(
            icao24=[AIRCRAFT_ID]
        )

        if states is None or not states.states:
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | "
                "No OpenSky data. No observation recorded."
            )
            continue

        # Confirm that the response belongs to our selected aircraft.
        state = next(
            (
                item
                for item in states.states
                if item.icao24
                and item.icao24.lower() == AIRCRAFT_ID
            ),
            None,
        )

        if state is None:
            print("Selected aircraft not present in response.")
            continue

        if (
            state.latitude is None
            or state.longitude is None
            or state.baro_altitude is None
            or state.velocity is None
            or state.true_track is None
            or state.vertical_rate is None
        ):
            print("Incomplete aircraft data. Observation skipped.")
            continue

        if state.time_position is None:
            print("No position timestamp. Observation skipped.")
            continue

        observation_time = float(state.time_position)

        if (
            last_observation_timestamp is not None
            and observation_time <= last_observation_timestamp
        ):
            print("Duplicate or older position update skipped.")
            continue

        last_observation_timestamp = observation_time

        timestamp = datetime.fromtimestamp(
            observation_time,
            timezone.utc
        ).isoformat()

        record = {
            "timestamp": timestamp,
            "timestamp_unix": observation_time,
            "icao24": state.icao24,
            "callsign": (state.callsign or "").strip(),
            "latitude": state.latitude,
            "longitude": state.longitude,
            "baro_altitude": state.baro_altitude,
            "geo_altitude": state.geo_altitude,
            "velocity": float(state.velocity),
            "heading": float(state.true_track),
            "vertical_rate": float(state.vertical_rate),
            "on_ground": state.on_ground,
            "squawk": state.squawk,
        }

        recording.append(record)

        # NEW: keep forwarding subsequent observations after a trigger.
        append_post_trigger_telemetry(record)

        ai_history.append(record)

        # Retain observations within the rolling 45-minute period.
        cutoff = observation_time - RECORDING_SECONDS

        while (
            recording
            and recording[0]["timestamp_unix"] < cutoff
        ):
            recording.popleft()

        print(
            f"{timestamp[11:19]} UTC | "
            f"Recorded {len(recording)} observations | "
            f"AI history {len(ai_history)}/{AI_HISTORY_SIZE} | "
            f"ALT {state.baro_altitude:.1f} m | "
            f"SPEED {state.velocity:.1f} m/s | "
            f"HDG {state.true_track:.1f}° | "
            f"VR {state.vertical_rate:.2f} m/s"
        )

        run_ai()

    except KeyboardInterrupt:
        print("\nAeroGuard monitoring stopped.")
        break

    except Exception as exc:
        print(f"\nOpenSky/API error: {exc}")
        print("Previous recording preserved. Retrying.")
        time.sleep(COLLECTION_INTERVAL)