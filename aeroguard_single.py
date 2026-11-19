import os
import csv
import time
import math
import shutil
import tempfile
import msvcrt
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from opensky_api import OpenSkyApi, TokenManager
from transmitter import activate_transmitter


# ============================================================
# AEROGUARD CONFIGURATION
# ============================================================

MODEL_FILE = Path("data/aeroguard_temporal_candidate.pkl")

COLLECTION_INTERVAL = 10
SNAPSHOT_INTERVAL_SECONDS = 60
RECORDING_MINUTES = 45
RECORDING_SECONDS = RECORDING_MINUTES * 60

AI_WINDOW = 10
ANOMALY_REQUIRED = 5

# A gap larger than this is reported as a telemetry gap.
MAX_EXPECTED_INTERVAL_SECONDS = 20

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
    "data/emergencies",
    "data/ground_station",
    "data/recordings",
):
    os.makedirs(folder, exist_ok=True)


# ============================================================
# LOAD CANDIDATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("                 AEROGUARD")
print("            CANDIDATE MODEL TEST")
print("=" * 60)

try:
    candidate = joblib.load(MODEL_FILE)

    if not isinstance(candidate, dict):
        raise ValueError("Model artifact must be a dictionary.")

    model = candidate["model"]
    MODEL_THRESHOLD = float(candidate["threshold"])
    candidate_features = list(candidate["features"])

    if candidate_features != FEATURE_NAMES:
        raise ValueError("Candidate feature order mismatch.")

    if not 0 <= MODEL_THRESHOLD <= 1:
        raise ValueError("Invalid model threshold.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba().")

    if 1 not in list(model.classes_):
        raise ValueError("Model has no emergency class 1.")

    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(
        FEATURE_NAMES
    ):
        raise ValueError("Model feature count mismatch.")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        if list(model_features) != FEATURE_NAMES:
            raise ValueError("Model feature names/order mismatch.")

except Exception as exc:
    raise SystemExit(f"Candidate model validation failed: {exc}")

print("Candidate model loaded.")
print("Features:", len(FEATURE_NAMES))
print(f"Threshold: {MODEL_THRESHOLD:.2%}")
print("Research prototype only. Not certified aviation equipment.")


# ============================================================
# OPENSKY CONNECTION
# ============================================================

try:
    token_manager = TokenManager.from_json_file("credentials.json")
    api = OpenSkyApi(token_manager=token_manager)
except Exception as exc:
    raise SystemExit(f"Could not configure OpenSky: {exc}")


# ============================================================
# FIND AIRCRAFT
# ============================================================

def find_aircraft():
    try:
        response = api.get_states()

        if response is None or not response.states:
            return []

        results = []

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

            if (
                all(value is not None for value in required)
                and state.on_ground is False
            ):
                results.append(state)

        return results[:10]

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
        f"ALT {altitude:.0f} m SPEED {state.velocity:.1f} m/s"
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
last_ai_window_end = None
last_snapshot_time = time.monotonic()

anomaly_count = 0
emergency_triggered = False
test_emergency = False


# ============================================================
# RECORDING QUALITY CHECKS
# ============================================================

def recording_report(rows):
    """Return measured coverage, duplicates, and telemetry gaps."""

    if not rows:
        return {
            "valid": False,
            "coverage_seconds": 0,
            "coverage_minutes": 0,
            "duplicates": 0,
            "gaps": [],
            "full": False,
        }

    timestamps = []

    for row in rows:
        try:
            value = float(row["timestamp_unix"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("A recording row has an invalid timestamp.")

        if not math.isfinite(value):
            raise ValueError("A recording timestamp is not finite.")

        timestamps.append(value)

    if any(
        timestamps[i] < timestamps[i - 1]
        for i in range(1, len(timestamps))
    ):
        raise ValueError("Recording timestamps are not chronological.")

    differences = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    gaps = [
        {
            "after": timestamps[i],
            "seconds": differences[i],
        }
        for i in range(len(differences))
        if differences[i] > MAX_EXPECTED_INTERVAL_SECONDS
    ]

    coverage = timestamps[-1] - timestamps[0]

    return {
        "valid": True,
        "coverage_seconds": coverage,
        "coverage_minutes": coverage / 60,
        "duplicates": len(timestamps) - len(set(timestamps)),
        "gaps": gaps,
        "full": coverage >= RECORDING_SECONDS,
    }


def write_recording(filename):
    """Save a recording atomically and verify the written CSV."""

    if not recording:
        print("ERROR: No observations available to save.")
        return None

    rows = [dict(row) for row in recording]
    temp_filename = None

    try:
        report = recording_report(rows)
        df = pd.DataFrame(rows)

        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="aeroguard_tmp_",
            dir=str(destination.parent),
            newline="",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_filename = temp_file.name
            df.to_csv(temp_file, index=False)

        verified = pd.read_csv(temp_filename)

        if len(verified) != len(rows):
            raise ValueError("Saved CSV row count mismatch.")

        saved_timestamps = pd.to_numeric(
            verified["timestamp_unix"], errors="coerce"
        )

        original_timestamps = pd.to_numeric(
            df["timestamp_unix"], errors="coerce"
        )

        if saved_timestamps.isna().any():
            raise ValueError("Saved CSV contains invalid timestamps.")

        if not np.array_equal(
            original_timestamps.to_numpy(dtype=float),
            saved_timestamps.to_numpy(dtype=float),
        ):
            raise ValueError("Saved CSV timestamps differ from source.")

        # Replace destination only after the temporary CSV passes checks.
        os.replace(temp_filename, destination)
        temp_filename = None

        print("\n========== RECORDING REPORT ==========")
        print("File:", destination.resolve())
        print("Observations:", len(rows))
        print(f"Coverage: {report['coverage_minutes']:.2f} minutes")
        print("Duplicate timestamps:", report["duplicates"])
        print("CSV read-back verification: PASSED")

        if report["full"]:
            print("45-minute coverage: COMPLETE")
        else:
            short_by = (
                RECORDING_SECONDS - report["coverage_seconds"]
            ) / 60
            print("45-minute coverage: PARTIAL")
            print(f"Short by: {max(0, short_by):.2f} minutes")

        if report["gaps"]:
            print("Telemetry gaps detected:", len(report["gaps"]))
            for gap in report["gaps"][:10]:
                print(f"  Gap: {gap['seconds']:.1f} seconds")
            if len(report["gaps"]) > 10:
                print("  Additional gaps omitted from display.")
        else:
            print("Telemetry gaps above threshold: 0")

        if report["duplicates"]:
            print("WARNING: Duplicate timestamps found.")

        print("======================================\n")
        return str(destination)

    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Recording save failed: {exc}")
        return None

    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError:
                print("Warning: Could not remove temporary file.")


def save_previous_recording():
    return write_recording(
        Path("data/emergencies")
        / AIRCRAFT_ID
        / "previous_45_minutes.csv"
    )


def save_rolling_snapshot():
    return write_recording(
        Path("data/recordings")
        / AIRCRAFT_ID
        / "rolling_45_minutes.csv"
    )


# ============================================================
# VERIFY GROUND-STATION COPY
# ============================================================

def verify_ground_station_copy(source_file):
    destination = (
        Path("data/ground_station")
        / AIRCRAFT_ID
        / "previous_45_minutes.csv"
    )

    try:
        if not source_file or not Path(source_file).is_file():
            print("Ground-station verification failed: source missing.")
            return False

        if not destination.is_file():
            print("Ground-station verification failed: copy missing.")
            return False

        source_df = pd.read_csv(source_file)
        destination_df = pd.read_csv(destination)

        if list(source_df.columns) != list(destination_df.columns):
            print("Ground-station verification failed: columns differ.")
            return False

        if len(source_df) != len(destination_df):
            print("Ground-station verification failed: row count differs.")
            return False

        if not source_df.equals(destination_df):
            print("Ground-station verification failed: data differs.")
            return False

        print("Ground-station CSV verification: PASSED")
        return True

    except Exception as exc:
        print(f"Ground-station verification failed: {exc}")
        return False


# ============================================================
# POST-TRIGGER TELEMETRY
# ============================================================

def append_post_trigger_telemetry(record):
    if not emergency_triggered:
        return

    filename = (
        Path("data/ground_station")
        / AIRCRAFT_ID
        / "post_trigger_telemetry.csv"
    )

    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        needs_header = (
            not filename.exists() or filename.stat().st_size == 0
        )

        with filename.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(record.keys())
            )
            if needs_header:
                writer.writeheader()
            writer.writerow(record)

    except (OSError, ValueError, TypeError) as exc:
        print("Post-trigger logging failed:", exc)


# ============================================================
# EMERGENCY HANDLER
# ============================================================

def trigger_emergency(issue):
    global emergency_triggered

    if emergency_triggered:
        return

    emergency_triggered = True

    print("\n" + "=" * 60)
    print("AEROGUARD SIMULATED EMERGENCY")
    print("=" * 60)
    print("Issue:", issue)

    recording_file = save_previous_recording()
    issues = [issue]

    if recording_file:
        try:
            rows = list(recording)
            report = recording_report(rows)

            if not report["full"]:
                issues.append(
                    "WARNING: Available recording is PARTIAL, not a "
                    "complete 45-minute history."
                )

            if report["gaps"]:
                issues.append(
                    f"Recording contains {len(report['gaps'])} "
                    "telemetry gap(s) exceeding the configured limit."
                )

            if report["duplicates"]:
                issues.append(
                    f"Recording contains {report['duplicates']} "
                    "duplicate timestamp(s)."
                )

        except Exception as exc:
            issues.append(f"Recording quality check failed: {exc}")

    else:
        issues.append(
            "Recording unavailable or save failed. "
            "Flight history could not be confirmed."
        )

    try:
        result = activate_transmitter(
            aircraft_id=AIRCRAFT_ID,
            callsign=CALLSIGN,
            issues=issues,
            recording=recording_file,
        )

        print("Transmitter result:", "SUCCESS" if result else "FAILED")

        if result and recording_file:
            copy_ok = verify_ground_station_copy(recording_file)
            print(
                "Ground-station verification:",
                "PASSED" if copy_ok else "FAILED",
            )

    except Exception as exc:
        print("Transmitter error:", exc)

    print("Simulation only. No real ATC connection.")


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


def trigger_test_emergency():
    global test_emergency

    if test_emergency and not emergency_triggered:
        test_emergency = False
        trigger_emergency("TEST EMERGENCY")


# ============================================================
# FEATURE ENGINEERING
# Keep this feature order aligned with the trained model.
# ============================================================

def heading_statistics(values):
    radians = np.deg2rad(np.asarray(values, dtype=float))

    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))

    mean = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
    resultant = np.clip(
        np.sqrt(sin_mean ** 2 + cos_mean ** 2),
        1e-12,
        1.0,
    )
    std = np.rad2deg(np.sqrt(-2 * np.log(resultant)))

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

    altitude = df["baro_altitude"].to_numpy(dtype=float)
    velocity = df["velocity"].to_numpy(dtype=float)
    heading = df["heading"].to_numpy(dtype=float)
    vertical = df["vertical_rate"].to_numpy(dtype=float)
    times = df["timestamp_unix"].to_numpy(dtype=float)

    combined = np.concatenate(
        [altitude, velocity, heading, vertical, times]
    )
    if not np.all(np.isfinite(combined)):
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
        "vertical_rate_nonzero_fraction": float(
            np.mean(np.abs(vertical) > 0.5)
        ),
    }

    result = pd.DataFrame([features], columns=FEATURE_NAMES)

    if result.isna().any().any():
        raise ValueError("Feature extraction produced missing values.")

    return result


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

    window = list(ai_history)
    ai_history.clear()

    window_end = window[-1]["timestamp_unix"]

    if last_ai_window_end == window_end:
        return

    last_ai_window_end = window_end

    try:
        features = create_features(window)
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        score = float(probabilities[classes.index(1)])
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
            print(
                f"Qualifying windows: "
                f"{anomaly_count}/{ANOMALY_REQUIRED}"
            )
        else:
            anomaly_count = 0
            print("Qualifying anomaly streak reset.")

        if anomaly_count >= ANOMALY_REQUIRED:
            trigger_emergency(
                "Candidate model detected abnormal flight behaviour"
            )

    except Exception as exc:
        anomaly_count = 0
        print("Candidate AI evaluation failed:", exc)


# ============================================================
# PROCESS TELEMETRY
# ============================================================

def process_response(response):
    global last_observation_timestamp, last_snapshot_time

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

    if last_observation_timestamp is not None:
        interval = observation_time - last_observation_timestamp
        if interval > MAX_EXPECTED_INTERVAL_SECONDS:
            print(
                f"TELEMETRY GAP: {interval:.1f} seconds "
                "since the previous accepted observation."
            )

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
    while recording and recording[0]["timestamp_unix"] < cutoff:
        recording.popleft()

    print(
        f"{timestamp[11:19]} UTC | "
        f"Recorded {len(recording)} observations | "
        f"AI batch {len(ai_history)}/{AI_WINDOW} | "
        f"ALT {state.baro_altitude:.1f} m | "
        f"SPEED {state.velocity:.1f} m/s"
    )

    now = time.monotonic()
    if now - last_snapshot_time >= SNAPSHOT_INTERVAL_SECONDS:
        save_rolling_snapshot()
        last_snapshot_time = now

    try:
        run_ai()
    finally:
        append_post_trigger_telemetry(record)

    return True


# ============================================================
# MAIN LOOP
# ============================================================

print("\nStarting AeroGuard candidate monitoring...")
print("Aircraft:", AIRCRAFT_ID)
print("Recording window: 45 minutes")
print("Collection interval:", COLLECTION_INTERVAL, "seconds")
print("AI window:", AI_WINDOW, "non-overlapping observations")
print(f"Candidate threshold: {MODEL_THRESHOLD:.2%}")
print("Press E to trigger a TEST EMERGENCY.")
print("Press Ctrl+C to stop.")
print("Research testing only. No real ATC connection.")
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
            print("OpenSky request failed:", exc)
            continue

        try:
            process_response(response)
        except Exception as exc:
            print("Telemetry processing failed:", exc)

except KeyboardInterrupt:
    print("\nStopping AeroGuard candidate...")
    save_rolling_snapshot()
    print("Monitoring stopped.")

except Exception as exc:
    print("\nAeroGuard stopped unexpectedly:", exc)