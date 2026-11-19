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
# AEROGUARD CONFIGURATION
# ============================================================

MODEL_FILE = "aeroguard_model_10.pkl"

COLLECTION_INTERVAL = 10
SNAPSHOT_INTERVAL_SECONDS = 60
RECORDING_MINUTES = 45
RECORDING_SECONDS = RECORDING_MINUTES * 60

AI_WINDOW = 10
ANOMALY_REQUIRED = 5
EMERGENCY_CONFIDENCE = 85.0

FEATURE_NAMES = [
    "altitude_mean",
    "altitude_std",
    "altitude_change",
    "velocity_mean",
    "velocity_std",
    "velocity_change",
    "heading_mean",
    "heading_std",
    "vertical_rate_mean",
    "vertical_rate_std",
    "vertical_rate_min",
    "vertical_rate_max",
]

os.makedirs("data/emergencies", exist_ok=True)
os.makedirs("data/ground_station", exist_ok=True)
os.makedirs("data/recordings", exist_ok=True)

print("\n" + "=" * 60)
print("             AEROGUARD")
print("       SINGLE AIRCRAFT MONITOR")
print("=" * 60)


# ============================================================
# LOAD MODEL AND CONNECT TO OPENSKY
# ============================================================

print("\nLoading AI model...")

try:
    model = joblib.load(MODEL_FILE)

    # Fail early if the loaded model expects different features.
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is not None:
        if list(model_features) != FEATURE_NAMES:
            raise ValueError(
                "Model feature names/order do not match this script.\n"
                f"Model expects: {list(model_features)}\n"
                f"Script provides: {FEATURE_NAMES}"
            )

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba().")

except Exception as exc:
    raise SystemExit(f"Could not load/validate AI model: {exc}")

print("AI model loaded.")

try:
    token_manager = TokenManager.from_json_file("credentials.json")
    api = OpenSkyApi(token_manager=token_manager)
except Exception as exc:
    raise SystemExit(f"Could not configure OpenSky: {exc}")

print("OpenSky connection configured.")


# ============================================================
# FIND AND SELECT AIRCRAFT
# ============================================================

def find_aircraft():
    try:
        response = api.get_states()

        if response is None or not response.states:
            return []

        result = []

        for state in response.states:
            if (
                state.icao24
                and state.latitude is not None
                and state.longitude is not None
                and state.baro_altitude is not None
                and state.velocity is not None
                and state.true_track is not None
                and state.vertical_rate is not None
                and state.time_position is not None
                and state.on_ground is False
            ):
                result.append(state)

        return result[:10]

    except Exception as exc:
        print(f"OpenSky search error: {exc}")
        return []


print("\nSearching for aircraft...")
aircraft_list = []

while not aircraft_list:
    aircraft_list = find_aircraft()

    if not aircraft_list:
        print("No aircraft found. Retrying in 10 seconds.")
        time.sleep(10)

print("\nAVAILABLE AIRCRAFT\n")

for index, state in enumerate(aircraft_list, start=1):
    callsign = (state.callsign or "").strip() or "UNKNOWN"

    altitude = (
        state.geo_altitude
        if state.geo_altitude is not None
        else state.baro_altitude
    )

    print(
        f"{index}. {state.icao24.lower()} {callsign} "
        f"ALT {altitude:.0f} m "
        f"SPEED {state.velocity:.1f} m/s"
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

print("\nSelected aircraft:", AIRCRAFT_ID)
print("Callsign:", CALLSIGN)


# ============================================================
# MONITOR STATE
# ============================================================

recording = deque()
ai_history = deque()

last_observation_timestamp = None
last_ai_window_end = None
last_snapshot_time = time.monotonic()

anomaly_count = 0
emergency_triggered = False
test_emergency = False


# ============================================================
# KEYBOARD COMMAND
# ============================================================

def check_test_command():
    global test_emergency

    while msvcrt.kbhit():
        key = msvcrt.getwch()

        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue

        if key.lower() == "e" and not emergency_triggered:
            test_emergency = True
            print("\nTEST EMERGENCY REQUESTED.")


# ============================================================
# RECORDING SAVE FUNCTIONS
# ============================================================

def write_recording(filename):
    if not recording:
        print("No observations available to save.")
        return None

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(list(recording))

        # Keep the human-readable timestamp and telemetry fields.
        df = df.drop(columns=["timestamp_unix"], errors="ignore")
        df.to_csv(filename, index=False)

        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            print("ERROR: Recording file is empty.")
            return None

        timestamps = [
            row["timestamp_unix"]
            for row in recording
        ]

        coverage = (
            (max(timestamps) - min(timestamps)) / 60
            if len(timestamps) > 1 else 0
        )

        print(f"Saved recording: {filename}")
        print(f"Observations: {len(recording)}")
        print(f"Actual time coverage: {coverage:.1f} minutes")

        if coverage < RECORDING_MINUTES - 1:
            print("WARNING: This is not a complete 45-minute recording.")

        return filename

    except (OSError, ValueError, TypeError) as exc:
        print(f"Recording save failed: {exc}")
        return None


def save_previous_recording():
    filename = os.path.join(
        "data", "emergencies", AIRCRAFT_ID, "previous_45_minutes.csv"
    )
    return write_recording(filename)


def save_rolling_snapshot():
    filename = os.path.join(
        "data", "recordings", AIRCRAFT_ID, "rolling_45_minutes.csv"
    )
    return write_recording(filename)


# ============================================================
# SIMULATED GROUND-STATION TELEMETRY
# ============================================================

def append_post_trigger_telemetry(record):
    if not emergency_triggered:
        return

    filename = os.path.join(
        "data", "ground_station", AIRCRAFT_ID,
        "post_trigger_telemetry.csv"
    )

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        needs_header = (
            not os.path.isfile(filename)
            or os.path.getsize(filename) == 0
        )

        with open(
            filename, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(record.keys())
            )

            if needs_header:
                writer.writeheader()

            writer.writerow(record)

    except (OSError, ValueError, TypeError) as exc:
        print(f"Post-trigger logging failed: {exc}")


# ============================================================
# EMERGENCY HANDLER
# ============================================================

def trigger_emergency(issue):
    global emergency_triggered

    if emergency_triggered:
        return

    # Latch immediately to prevent duplicate triggers.
    emergency_triggered = True

    print("\n" + "=" * 60)
    print("AEROGUARD EMERGENCY CONDITION")
    print("=" * 60)
    print("Issue:", issue)

    recording_file = save_previous_recording()
    issues = [issue]

    if recording_file is None:
        issues.append(
            "Recording unavailable or save failed. "
            "The emergency report may not include flight history."
        )

    try:
        result = activate_transmitter(
            aircraft_id=AIRCRAFT_ID,
            callsign=CALLSIGN,
            issues=issues,
            recording=recording_file,
        )

        print(
            "Transmitter result:",
            "SUCCESS" if result else "FAILED"
        )

    except Exception as exc:
        print(f"Transmitter error: {exc}")

    print("Simulation only. No real ATC connection.")


def trigger_test_emergency():
    global test_emergency

    if not test_emergency or emergency_triggered:
        return

    test_emergency = False
    trigger_emergency("TEST EMERGENCY")


# ============================================================
# MODEL FEATURE CREATION
# ============================================================

def create_features(history):
    df = pd.DataFrame(history)

    # These calculations match the training builder's
    # NumPy population standard deviation (ddof=0).
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

    # Enforce training feature order.
    return pd.DataFrame([features], columns=FEATURE_NAMES)


# ============================================================
# AI EVALUATION
# ============================================================

def run_ai():
    global anomaly_count, last_ai_window_end

    if len(ai_history) < AI_WINDOW:
        print(
            f"AI waiting: {len(ai_history)}/{AI_WINDOW} "
            "observations in current window."
        )
        return

    # Copy the completed window. Clear it so the next prediction
    # uses a new, non-overlapping batch, as in model training.
    window = list(ai_history)
    ai_history.clear()

    window_end = window[-1]["timestamp_unix"]

    if last_ai_window_end == window_end:
        return

    last_ai_window_end = window_end

    try:
        features = create_features(window)

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        emergency_index = (
            classes.index(1) if 1 in classes else None
        )

        score = (
            float(probabilities[emergency_index]) * 100
            if emergency_index is not None else 0.0
        )

        span = (
            window[-1]["timestamp_unix"]
            - window[0]["timestamp_unix"]
        )

        print("\nAEROGUARD AI RESULT")
        print("Prediction:", "ANOMALY" if prediction == 1 else "NORMAL")
        print(f"Model score: {score:.2f}%")
        print(f"Window time span: {span:.1f} seconds")

        if (
            prediction == 1
            and score >= EMERGENCY_CONFIDENCE
        ):
            anomaly_count += 1
            print(
                f"Qualifying windows: "
                f"{anomaly_count}/{ANOMALY_REQUIRED}"
            )
        else:
            anomaly_count = 0
            print("Qualifying anomaly streak reset.")

        if anomaly_count >= ANOMALY_REQUIRED:
            trigger_emergency(
                "AI detected abnormal flight behaviour"
            )

    except Exception as exc:
        anomaly_count = 0
        print(f"AI evaluation failed: {exc}")


# ============================================================
# MAIN MONITORING LOOP
# ============================================================

print("\nStarting AeroGuard monitoring...")
print("Aircraft:", AIRCRAFT_ID)
print("Recording window: 45 minutes")
print("Collection interval:", COLLECTION_INTERVAL, "seconds")
print("AI window:", AI_WINDOW, "non-overlapping observations")
print("Press E to trigger a TEST EMERGENCY.")
print("Press Ctrl+C to stop.")
print("Research prototype only. No real ATC connection.")
print("=" * 60)

next_poll = time.monotonic()

try:
    while True:
        check_test_command()
        trigger_test_emergency()

        now = time.monotonic()

        if now < next_poll:
            time.sleep(0.1)
            continue

        next_poll = time.monotonic() + COLLECTION_INTERVAL

        try:
            response = api.get_states(icao24=[AIRCRAFT_ID])
        except Exception as exc:
            print(f"OpenSky request failed: {exc}")
            continue

        if response is None or not response.states:
            print("No OpenSky observation. Existing recording retained.")
            continue

        state = next(
            (
                item for item in response.states
                if item.icao24
                and item.icao24.lower() == AIRCRAFT_ID
            ),
            None,
        )

        if state is None:
            print("Selected aircraft not present in response.")
            continue

        required = (
            state.latitude,
            state.longitude,
            state.baro_altitude,
            state.velocity,
            state.true_track,
            state.vertical_rate,
            state.time_position,
        )

        if any(value is None for value in required):
            print("Incomplete telemetry. Observation skipped.")
            continue

        observation_time = float(state.time_position)
        current_time = datetime.now(timezone.utc).timestamp()
        age = current_time - observation_time

        if age > 60 or age < -10:
            print(f"Stale/future observation skipped. Age: {age:.1f}s")
            continue

        if (
            last_observation_timestamp is not None
            and observation_time <= last_observation_timestamp
        ):
            print("Duplicate or older observation skipped.")
            continue

        last_observation_timestamp = observation_time

        timestamp = datetime.fromtimestamp(
            observation_time, timezone.utc
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
        ai_history.append(record)

        cutoff = observation_time - RECORDING_SECONDS

        while (
            recording
            and recording[0]["timestamp_unix"] < cutoff
        ):
            recording.popleft()

        print(
            f"{timestamp[11:19]} UTC | "
            f"Recorded {len(recording)} observations | "
            f"AI batch {len(ai_history)}/{AI_WINDOW} | "
            f"ALT {state.baro_altitude:.1f} m | "
            f"SPEED {state.velocity:.1f} m/s"
        )

        # Save snapshots without overwriting emergency evidence.
        snapshot_now = time.monotonic()

        if (
            snapshot_now - last_snapshot_time
            >= SNAPSHOT_INTERVAL_SECONDS
        ):
            save_rolling_snapshot()
            last_snapshot_time = snapshot_now

        try:
            run_ai()
        finally:
            append_post_trigger_telemetry(record)

except KeyboardInterrupt:
    print("\nStopping AeroGuard...")
    save_rolling_snapshot()
    print("Monitoring stopped.")

except Exception as exc:
    print(f"\nAeroGuard stopped due to an unexpected error: {exc}")
    save_rolling_snapshot()