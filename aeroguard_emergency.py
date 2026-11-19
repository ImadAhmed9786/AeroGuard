import os
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from opensky_api import OpenSkyApi, TokenManager
from transmitter import activate_transmitter


# ==========================================
# SETTINGS
# ==========================================

COLLECTION_INTERVAL = 10

AI_WINDOW = 20

RECORDING_MINUTES = 45
QUEUE_SIZE = (RECORDING_MINUTES * 60) // COLLECTION_INTERVAL

# Safer emergency trigger
REQUIRED_ANOMALIES = 5
MIN_EMERGENCY_CONFIDENCE = 85

MODEL_FILE = "aeroguard_model.pkl"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading AeroGuard AI...", flush=True)

model = joblib.load(MODEL_FILE)

print("AI model loaded.", flush=True)


# ==========================================
# OPENSKY
# ==========================================

token_manager = TokenManager.from_json_file("credentials.json")

api = OpenSkyApi(token_manager=token_manager)


# ==========================================
# PER-AIRCRAFT MEMORY
# ==========================================

recording = defaultdict(
    lambda: deque(maxlen=QUEUE_SIZE)
)

ai_history = defaultdict(
    lambda: deque(maxlen=AI_WINDOW)
)

anomaly_count = defaultdict(int)

emergency_triggered = set()


# ==========================================
# START
# ==========================================

print()
print("==========================================")
print("          AEROGUARD LIVE SYSTEM")
print("==========================================")
print(f"AI window: {AI_WINDOW} observations")
print(f"Recording buffer: {QUEUE_SIZE} observations")
print(f"Recording duration: {RECORDING_MINUTES} minutes")
print(f"Required consecutive anomalies: {REQUIRED_ANOMALIES}")
print(f"Minimum confidence: {MIN_EMERGENCY_CONFIDENCE}%")
print("==========================================")
print()
print("Press CTRL+C to stop.\n")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    cycle_start = time.time()

    try:

        print("Collecting OpenSky data...", flush=True)


        # ==================================
        # GET LIVE DATA
        # ==================================

        try:

            states = api.get_states()

        except Exception as e:

            print(
                f"OpenSky request failed: {e}",
                flush=True
            )

            print(
                "Skipping this cycle...",
                flush=True
            )

            time.sleep(COLLECTION_INTERVAL)

            continue


        if states is None or not states.states:

            print(
                "No aircraft data received.",
                flush=True
            )

            time.sleep(COLLECTION_INTERVAL)

            continue


        timestamp = pd.Timestamp.now("UTC")


        aircraft_count = 0
        valid_count = 0

        ready_aircraft = []
        feature_rows = []


        # ==================================
        # COLLECT AIRCRAFT DATA
        # ==================================

        for aircraft in states.states:

            aircraft_id = aircraft.icao24

            if aircraft_id is None:
                continue


            aircraft_count += 1


            # --------------------------------
            # REQUIRED VALUES
            # --------------------------------

            if (
                aircraft.baro_altitude is None
                or aircraft.velocity is None
                or aircraft.true_track is None
            ):

                continue


            valid_count += 1


            vertical_rate = aircraft.vertical_rate

            if vertical_rate is None:
                vertical_rate = 0.0


            # ==================================
            # SAVE LIVE RECORDING
            # ==================================

            record = {
                "timestamp": timestamp,
                "icao24": aircraft_id,
                "callsign": aircraft.callsign,
                "longitude": aircraft.longitude,
                "latitude": aircraft.latitude,
                "baro_altitude": aircraft.baro_altitude,
                "velocity": aircraft.velocity,
                "heading": aircraft.true_track,
                "vertical_rate": vertical_rate,
                "geo_altitude": aircraft.geo_altitude,
                "on_ground": aircraft.on_ground,
                "squawk": aircraft.squawk
            }

            recording[aircraft_id].append(record)


            # ==================================
            # AI HISTORY
            # ==================================

            ai_history[aircraft_id].append(
                {
                    "baro_altitude": aircraft.baro_altitude,
                    "velocity": aircraft.velocity,
                    "heading": aircraft.true_track,
                    "vertical_rate": vertical_rate
                }
            )


            # ==================================
            # WAIT FOR 20 OBSERVATIONS
            # ==================================

            if len(ai_history[aircraft_id]) < AI_WINDOW:

                continue


            ready_aircraft.append(aircraft_id)


            # ==================================
            # BUILD FEATURES
            # ==================================

            history = ai_history[aircraft_id]


            altitude = np.array(
                [x["baro_altitude"] for x in history],
                dtype=float
            )

            velocity = np.array(
                [x["velocity"] for x in history],
                dtype=float
            )

            heading = np.array(
                [x["heading"] for x in history],
                dtype=float
            )

            vertical_rate = np.array(
                [x["vertical_rate"] for x in history],
                dtype=float
            )


            feature_rows.append(
                [
                    np.mean(altitude),
                    np.std(altitude),
                    altitude[-1] - altitude[0],

                    np.mean(velocity),
                    np.std(velocity),
                    velocity[-1] - velocity[0],

                    np.mean(heading),
                    np.std(heading),

                    np.mean(vertical_rate),
                    np.std(vertical_rate),
                    np.min(vertical_rate),
                    np.max(vertical_rate)
                ]
            )


        # ==================================
        # BATCH AI PREDICTION
        # ==================================

        predictions = 0
        anomalies = 0


        if feature_rows:

            feature_matrix = np.asarray(
                feature_rows,
                dtype=float
            )


            model_predictions = model.predict(
                feature_matrix
            )

            model_probabilities = model.predict_proba(
                feature_matrix
            )


            predictions = len(model_predictions)


            # ==================================
            # HANDLE EACH AIRCRAFT RESULT
            # ==================================

            for i, aircraft_id in enumerate(
                ready_aircraft
            ):

                prediction = model_predictions[i]

                confidence = (
                    model_probabilities[i][1] * 100
                )


                # ==================================
                # NORMAL
                # ==================================

                if prediction == 0:

                    anomaly_count[aircraft_id] = 0

                    continue


                # ==================================
                # ANOMALY
                # ==================================

                anomalies += 1

                anomaly_count[aircraft_id] += 1


                print(
                    f"AI anomaly: "
                    f"{aircraft_id} | "
                    f"Confidence: {confidence:.1f}% | "
                    f"Count: "
                    f"{anomaly_count[aircraft_id]}/"
                    f"{REQUIRED_ANOMALIES}",
                    flush=True
                )


                # ==================================
                # EMERGENCY CONDITION
                # ==================================

                if (
                    anomaly_count[aircraft_id]
                    >= REQUIRED_ANOMALIES
                    and confidence
                    >= MIN_EMERGENCY_CONFIDENCE
                    and aircraft_id
                    not in emergency_triggered
                ):

                    emergency_triggered.add(
                        aircraft_id
                    )


                    print()
                    print("==========================================")
                    print("🚨 AEROGUARD EMERGENCY DETECTED")
                    print("==========================================")

                    print(
                        f"Aircraft: {aircraft_id}"
                    )

                    print(
                        f"Confidence: "
                        f"{confidence:.2f}%"
                    )

                    print(
                        f"Consecutive anomalies: "
                        f"{anomaly_count[aircraft_id]}"
                    )


                    # ==================================
                    # SAVE PREVIOUS 45 MINUTES
                    # ==================================

                    emergency_folder = os.path.join(
                        "data",
                        "emergencies",
                        aircraft_id
                    )

                    os.makedirs(
                        emergency_folder,
                        exist_ok=True
                    )


                    emergency_file = os.path.join(
                        emergency_folder,
                        "previous_45_minutes.csv"
                    )


                    emergency_data = pd.DataFrame(
                        list(
                            recording[aircraft_id]
                        )
                    )


                    emergency_data.to_csv(
                        emergency_file,
                        index=False
                    )


                    print(
                        "Previous 45 minutes saved:"
                    )

                    print(
                        emergency_file
                    )


                    # ==================================
                    # TRANSMITTER
                    # ==================================

                    detected_issues = [
                        {
                            "issue":
                            "AI detected abnormal flight behaviour",

                            "severity":
                            round(confidence, 2)
                        }
                    ]


                    activate_transmitter(
                        aircraft_id=aircraft_id,
                        issues=detected_issues
                    )


                    print(
                        "Emergency recording continues..."
                    )

                    print(
                        "==========================================\n"
                    )


        # ==================================
        # PERFORMANCE
        # ==================================

        elapsed = time.time() - cycle_start

        print(
            f"[{timestamp}] "
            f"Aircraft: {aircraft_count} | "
            f"Valid: {valid_count} | "
            f"AI Ready: {len(ready_aircraft)} | "
            f"Predictions: {predictions} | "
            f"Anomalies: {anomalies} | "
            f"Cycle: {elapsed:.2f}s",
            flush=True
        )


        # ==================================
        # KEEP 10 SECOND INTERVAL
        # ==================================

        remaining = max(
            0,
            COLLECTION_INTERVAL - elapsed
        )

        time.sleep(remaining)


    # ==================================
    # CTRL+C
    # ==================================

    except KeyboardInterrupt:

        print()
        print("AeroGuard stopped.")

        break


    # ==================================
    # OTHER ERROR
    # ==================================

    except Exception as e:

        print(
            f"Unexpected error: {e}",
            flush=True
        )

        print(
            "AeroGuard will continue...",
            flush=True
        )

        time.sleep(5)