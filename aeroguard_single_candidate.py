
"""
AEROGUARD: TEMPORAL CANDIDATE MONITOR
Research prototype only. Not certified aviation equipment.
No real ATC connection. Emergency transmission must remain simulated.
"""

import os
import csv
import time
import msvcrt
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from opensky_api import OpenSkyApi, TokenManager
from transmitter import activate_transmitter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_FILE = os.path.join(DATA_DIR, "aeroguard_temporal_candidate.pkl")
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")

COLLECTION_INTERVAL = 10
SNAPSHOT_INTERVAL_SECONDS = 60
RECORDING_MINUTES = 45
RECORDING_SECONDS = RECORDING_MINUTES * 60

AI_WINDOW = 10
MAX_GAP_SECONDS = 30
ANOMALY_REQUIRED = 5

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
    "altitude_slope",
    "velocity_slope",
    "vertical_rate_slope",
    "vertical_rate_abs_mean",
    "vertical_rate_nonzero_fraction",
]

for folder in (
    "emergencies",
    "ground_station",
    "recordings",
):
    os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

print("\n" + "=" * 60)
print("                 AEROGUARD")
print("          TEMPORAL CANDIDATE MONITOR")
print("=" * 60)


# ============================================================
# LOAD AND VALIDATE MODEL
# ============================================================

if not os.path.isfile(MODEL_FILE):
    raise SystemExit(f"Candidate model not found: {MODEL_FILE}")

try:
    candidate = joblib.load(MODEL_FILE)

    if not isinstance(candidate, dict):
        raise ValueError("Model artifact must be a dictionary.")

    required_keys = {
        "model", "features", "threshold",
        "window_size", "max_gap_seconds",
    }

    missing = required_keys - set(candidate)
    if missing:
        raise ValueError(f"Missing model fields: {missing}")

    model = candidate["model"]
    MODEL_THRESHOLD = float(candidate["threshold"])

    if list(candidate["features"]) != FEATURE_NAMES:
        raise ValueError("Candidate feature names/order mismatch.")

    if int(candidate["window_size"]) != AI_WINDOW:
        raise ValueError("Candidate window size mismatch.")

    if int(candidate["max_gap_seconds"]) != MAX_GAP_SECONDS:
        raise ValueError("Candidate maximum-gap setting mismatch.")

    if not 0.0 < MODEL_THRESHOLD < 1.0:
        raise ValueError("Invalid probability threshold.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba().")

    if 1 not in list(model.classes_):
        raise ValueError("Model does not contain positive class 1.")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        if list(model_features) != FEATURE_NAMES:
            raise ValueError("Model's own feature order does not match.")

    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ValueError("Model expects a different feature count.")

except Exception as exc:
    raise SystemExit(f"Candidate model validation failed: {exc}")

print("Candidate model loaded.")
print(f"Features: {len(FEATURE_NAMES)}")
print(f"Threshold: {MODEL_THRESHOLD:.2%}")
print(f"Window size: {AI_WINDOW} observations")
print(f"Maximum telemetry gap: {MAX_GAP_SECONDS} seconds")
print("Status: RESEARCH CANDIDATE ONLY")


# ============================================================
# OPENSKY CONNECTION
# ============================================================

try:
    token_manager = TokenManager.from_json_file(CREDENTIALS_FILE)
    api = OpenSkyApi(token_manager=token_manager)
except Exception as exc:
    raise SystemExit(f"Could not configure OpenSky: {exc}")


# ============================================================
# AIRCRAFT DISCOVERY
# ============================================================

def find_aircraft():
    try:
        response = api.get_states()

        if response is None or not response.states:
            return []

        result = []

        for state in response.states:
            required = (
                state.icao24,
                state.latitude,
                state.longitude,
                state.baro_altitude,
                state.velocity,
                state.true_track,
                state.vertical_rate,
                state.time_position,
            )

            if any(value is None for value in required):
                continue

            if state.on_ground is not False:
                continue

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
        print("No suitable aircraft found. Retrying in 10 seconds.")
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
        choice = int(input(f"\nSelect aircraft (1-{len(aircraft_list)}): "))
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
last_snapshot_time = time.monotonic()

anomaly_count = 0
emergency_triggered = False
test_emergency = False


# ============================================================
# TEST KEYBOARD COMMAND
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
# RECORDING
# ============================================================

def write_recording(filename):
    if not recording:
        print("No observations available to save.")
        return None

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(list(recording))
        df = df.drop(columns=["timestamp_unix"], errors="ignore")
        df.to_csv(filename, index=False)

        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            print("ERROR: Recording file is empty.")
            return None

        timestamps = [row["timestamp_unix"] for row in recording]
        coverage = (
            (max(timestamps) - min(timestamps)) / 60
            if len(timestamps) > 1 else 0
        )

        print(f"Saved recording: {filename}")
        print(f"Observations: {len(recording)}")
        print(f"Actual time coverage: {coverage:.1f} minutes")

        if coverage < RECORDING_MINUTES - 1:
            print("WARNING: Recording does not cover the full 45 minutes.")

        return filename

    except (OSError, ValueError, TypeError) as exc:
        print(f"Recording save failed: {exc}")
        return None


def save_previous_recording():
    filename = os.path.join(
        DATA_DIR, "emergencies", AIRCRAFT_ID, "previous_45_minutes.csv"
    )
    return write_recording(filename)


def save_rolling_snapshot():
    filename = os.path.join(
        DATA_DIR, "recordings", AIRCRAFT_ID, "rolling_45_minutes.csv"
    )
    return write_recording(filename)


# ============================================================
# POST-TRIGGER TELEMETRY LOGGING
# ============================================================

def append_post_trigger_telemetry(record):
    if not emergency_triggered:
        return

    filename = os.path.join(
        DATA_DIR, "ground_station", AIRCRAFT_ID,
        "post_trigger_telemetry.csv",
    )

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        needs_header = (
            not os.path.isfile(filename)
            or os.path.getsize(filename) == 0
        )

        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(record.keys()))

            if needs_header:
                writer.writeheader()

            writer.writerow(record)

    except (OSError, ValueError, TypeError) as exc:
        print(f"Post-trigger logging failed: {exc}")


# ============================================================
# SIMULATED EMERGENCY HANDLER
# ============================================================

def trigger_emergency(issue):
    global emergency_triggered

    if emergency_triggered:
        return

    emergency_triggered = True

    print("\n" + "=" * 60)
    print("AEROGUARD RESEARCH ALERT")
    print("=" * 60)
    print("Issue:", issue)
    print("This is a model-generated research alert, not a verified emergency.")

    recording_file = save_previous_recording()
    issues = [issue]

    if recording_file is None:
        issues.append(
            "Recording unavailable or save failed; flight history may be missing."
        )

    # This call must remain connected to a local simulation only.
    try:
        result = activate_transmitter(
            aircraft_id=AIRCRAFT_ID,
            callsign=CALLSIGN,
            issues=issues,
            recording=recording_file,
        )

        print("Simulated transmitter result:", "SUCCESS" if result else "FAILED")

    except Exception as exc:
        print(f"Simulated transmitter error: {exc}")

    print("Simulation only. No real ATC connection.")


def trigger_test_emergency():
    global test_emergency

    if not test_emergency or emergency_triggered:
        return

    test_emergency = False
    trigger_emergency("MANUAL TEST EMERGENCY")


# ============================================================
# FEATURE ENGINEERING
# Must match evaluate_heldout_flights.py.
# ============================================================

def heading_statistics(values):
    radians = np.deg2rad(np.asarray(values, dtype=float))

    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))

    mean = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360

    resultant = np.clip(
        np.sqrt(sin_mean**2 + cos_mean**2),
        1e-12,
        1.0,
    )

    std = np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))

    return float(mean), float(std)


def calculate_slope(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    elapsed = times - times[0]

    if len(values) < 2 or elapsed[-1] <= 0:
        return 0.0

    return float(np.polyfit(elapsed, values, 1)[0])


def create_features(history):
    df = pd.DataFrame(history)

    if len(df) != AI_WINDOW:
        raise ValueError(
            f"Expected {AI_WINDOW} observations, received {len(df)}."
        )

    times = df["timestamp_unix"].to_numpy(dtype=float)

    if np.any(np.diff(times) <= 0):
        raise ValueError("Window timestamps are not strictly increasing.")

    if np.max(np.diff(times)) > MAX_GAP_SECONDS:
        raise ValueError("Window contains a telemetry gap exceeding 30 seconds.")

    altitude = df["baro_altitude"].to_numpy(dtype=float)
    velocity = df["velocity"].to_numpy(dtype=float)
    heading = df["heading"].to_numpy(dtype=float)
    vertical = df["vertical_rate"].to_numpy(dtype=float)

    if not np.all(np.isfinite(
        np.concatenate([altitude, velocity, heading, vertical, times])
    )):
        raise ValueError("Telemetry window contains non-finite values.")

    heading_mean, heading_std = heading_statistics(heading)

    features = {
        "altitude_mean": float(np.mean(altitude)),
        "altitude_std": float(np.std(altitude, ddof=1)),
        "altitude_change": float(altitude[-1] - altitude[0]),

        "velocity_mean": float(np.mean(velocity)),
        "velocity_std": float(np.std(velocity, ddof=1)),
        "velocity_change": float(velocity[-1] - velocity[0]),

        "heading_mean": heading_mean,
        "heading_std": heading_std,

        "vertical_rate_mean": float(np.mean(vertical)),
        "vertical_rate_std": float(np.std(vertical, ddof=1)),
        "vertical_rate_min": float(np.min(vertical)),
        "vertical_rate_max": float(np.max(vertical)),

        "altitude_slope": calculate_slope(altitude, times),
        "velocity_slope": calculate_slope(velocity, times),
        "vertical_rate_slope": calculate_slope(vertical, times),

        "vertical_rate_abs_mean": float(np.mean(np.abs(vertical))),

        # Match the candidate training feature definition.
        "vertical_rate_nonzero_fraction": float(
    np.mean(np.abs(vertical) > 0.5)
        ),
    }

    result = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    if result.isna().any().any():
        raise ValueError("Feature extraction produced missing values.")

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

def run_ai():
    global anomaly_count

    if len(ai_history) < AI_WINDOW:
        print(
            f"AI waiting: {len(ai_history)}/{AI_WINDOW} "
            "observations in current window."
        )
        return

    window = list(ai_history)
    ai_history.clear()

    try:
        features = create_features(window)

        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        emergency_index = classes.index(1)

        score = float(probabilities[emergency_index])
        prediction = int(score >= MODEL_THRESHOLD)

        span = (
            window[-1]["timestamp_unix"]
            - window[0]["timestamp_unix"]
        )

        print("\nAEROGUARD CANDIDATE AI RESULT")
        print("Prediction:", "ANOMALY" if prediction else "NORMAL")
        print(f"Emergency probability: {score:.2%}")
        print(f"Candidate threshold: {MODEL_THRESHOLD:.2%}")
        print(f"Window time span: {span:.1f} seconds")

        if prediction:
            anomaly_count += 1
            print(f"Positive windows in a row: {anomaly_count}/{ANOMALY_REQUIRED}")
        else:
            anomaly_count = 0
            print("Positive-window streak reset.")

        if anomaly_count >= ANOMALY_REQUIRED:
            trigger_emergency(
                "Candidate model flagged five consecutive windows"
            )

    except Exception as exc:
        anomaly_count = 0
        print(f"Candidate AI evaluation failed: {exc}")


# ============================================================
# TELEMETRY PROCESSING
# ============================================================

def process_response(response):
    global last_observation_timestamp
    global last_snapshot_time
    global anomaly_count

    if response is None or not response.states:
        print("No OpenSky observation. Existing recording retained.")
        return False

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
        return False

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
        return False

    observation_time = float(state.time_position)
    current_time = datetime.now(timezone.utc).timestamp()
    age = current_time - observation_time

    if age > 60 or age < -10:
        print(f"Stale/future observation skipped. Age: {age:.1f}s")
        return False

    if (
        last_observation_timestamp is not None
        and observation_time <= last_observation_timestamp
    ):
        print("Duplicate or older observation skipped.")
        return False

    # Reset the partial AI window after a large timestamp gap.
    if last_observation_timestamp is not None:
        gap = observation_time - last_observation_timestamp

        if gap > MAX_GAP_SECONDS:
            print(f"Telemetry gap: {gap:.1f}s. Resetting partial AI window.")
            ai_history.clear()
            anomaly_count = 0

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
        "baro_altitude": float(state.baro_altitude),
        "geo_altitude": state.geo_altitude,
        "velocity": float(state.velocity),
        "heading": float(state.true_track) % 360,
        "vertical_rate": float(state.vertical_rate),
        "on_ground": state.on_ground,
        "squawk": state.squawk,
    }

    recording.append(record)
    ai_history.append(record)

    cutoff = observation_time - RECORDING_SECONDS

    while recording and recording[0]["timestamp_unix"] < cutoff:
        recording.popleft()

    print(
        f"{timestamp[11:19]} UTC | "
        f"Recorded {len(recording)} observations | "
        f"AI batch {len(ai_history)}/{AI_WINDOW} | "
        f"ALT {state.baro_altitude:.1f} m | "
        f"SPEED {state.velocity:.1f} m/s"
    )

    snapshot_now = time.monotonic()

    if snapshot_now - last_snapshot_time >= SNAPSHOT_INTERVAL_SECONDS:
        save_rolling_snapshot()
        last_snapshot_time = snapshot_now

    try:
        run_ai()
    finally:
        append_post_trigger_telemetry(record)

    return True


# ============================================================
# MAIN MONITOR LOOP
# ============================================================

print("\nStarting AeroGuard candidate monitoring...")
print("Aircraft:", AIRCRAFT_ID)
print("Recording window: 45 minutes")
print("Collection interval:", COLLECTION_INTERVAL, "seconds")
print("AI window:", AI_WINDOW, "non-overlapping observations")
print(f"Candidate threshold: {MODEL_THRESHOLD:.2%}")
print("Press E to trigger a simulated TEST EMERGENCY.")
print("Press Ctrl+C to stop.")
print("RESEARCH TESTING ONLY. No real ATC connection.")
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

        try:
            process_response(response)
        except Exception as exc:
            print(f"Telemetry processing failed; monitoring continues: {exc}")

except KeyboardInterrupt:
    print("\nStopping AeroGuard candidate...")
    save_rolling_snapshot()
    print("Monitoring stopped.")

except Exception as exc:
    print(f"\nAeroGuard stopped due to an unexpected error: {exc}")
    save_rolling_snapshot()
