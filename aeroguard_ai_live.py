import time
from collections import defaultdict, deque

import joblib
import pandas as pd

from opensky_api import OpenSkyApi, TokenManager
from transmitter import activate_transmitter


COLLECTION_INTERVAL = 10
WINDOW_SIZE = 20
REQUIRED_ANOMALIES = 2
MODEL_FILE = "aeroguard_model.pkl"


print("Loading AeroGuard AI...")
model = joblib.load(MODEL_FILE)
print("AI model loaded.")


token_manager = TokenManager.from_json_file("credentials.json")
api = OpenSkyApi(token_manager=token_manager)


aircraft_history = defaultdict(
    lambda: deque(maxlen=WINDOW_SIZE)
)

anomaly_count = defaultdict(int)
emergency_triggered = set()

total_predictions = 0
normal_predictions = 0
anomaly_predictions = 0


print("\n======================================")
print("AEROGUARD AI LIVE SYSTEM")
print("======================================")
print("Monitoring aircraft...")
print("AI detection active.")
print("Emergency requires 2 consecutive detections.")
print("Prediction statistics active.")
print("History tracking active.")
print("Press CTRL+C to stop.")
print()


while True:

    try:

        states = api.get_states()

        if states is None:
            print("No aircraft data received.")
            time.sleep(COLLECTION_INTERVAL)
            continue

        timestamp = pd.Timestamp.now("UTC")

        valid_aircraft = 0
        ready_aircraft = 0
        maximum_history = 0

        for aircraft in states.states:

            if aircraft.icao24 is None:
                continue

            # Required values for the trained model
            if (
                aircraft.baro_altitude is None
                or aircraft.velocity is None
                or aircraft.true_track is None
            ):
                continue

            # Vertical rate can occasionally be missing
            vertical_rate = aircraft.vertical_rate

            if vertical_rate is None:
                vertical_rate = 0.0

            valid_aircraft += 1

            aircraft_id = aircraft.icao24

            record = {
                "timestamp": timestamp,
                "icao24": aircraft_id,
                "baro_altitude": aircraft.baro_altitude,
                "velocity": aircraft.velocity,
                "heading": aircraft.true_track,
                "vertical_rate": vertical_rate
            }

            aircraft_history[aircraft_id].append(record)

            history_length = len(
                aircraft_history[aircraft_id]
            )

            maximum_history = max(
                maximum_history,
                history_length
            )

            if history_length < WINDOW_SIZE:
                continue

            ready_aircraft += 1

            history = pd.DataFrame(
                aircraft_history[aircraft_id]
            )

            features = pd.DataFrame([{

                "altitude_mean":
                    history["baro_altitude"].mean(),

                "altitude_std":
                    history["baro_altitude"].std(),

                "altitude_change":
                    history["baro_altitude"].iloc[-1]
                    - history["baro_altitude"].iloc[0],

                "velocity_mean":
                    history["velocity"].mean(),

                "velocity_std":
                    history["velocity"].std(),

                "velocity_change":
                    history["velocity"].iloc[-1]
                    - history["velocity"].iloc[0],

                "heading_mean":
                    history["heading"].mean(),

                "heading_std":
                    history["heading"].std(),

                "vertical_rate_mean":
                    history["vertical_rate"].mean(),

                "vertical_rate_std":
                    history["vertical_rate"].std(),

                "vertical_rate_min":
                    history["vertical_rate"].min(),

                "vertical_rate_max":
                    history["vertical_rate"].max()

            }])

            features = features.fillna(0)

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            emergency_confidence = probabilities[1] * 100

            total_predictions += 1

            if prediction == 0:

                normal_predictions += 1
                anomaly_count[aircraft_id] = 0

            else:

                anomaly_predictions += 1
                anomaly_count[aircraft_id] += 1

            if (
                anomaly_count[aircraft_id]
                >= REQUIRED_ANOMALIES
                and aircraft_id not in emergency_triggered
            ):

                emergency_triggered.add(aircraft_id)

                print("\n")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("       AEROGUARD EMERGENCY")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"Aircraft: {aircraft_id}")
                print(f"Detection time: {timestamp}")
                print("AI classification: EMERGENCY")
                print(
                    f"AI confidence: "
                    f"{emergency_confidence:.2f}%"
                )
                print("Activating transmitter...")

                activate_transmitter(
                    aircraft_id=aircraft_id,
                    issues=[
                        {
                            "issue":
                                "Abnormal flight behaviour",
                            "severity":
                                round(emergency_confidence)
                        }
                    ]
                )

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print()

        print(
            f"[{timestamp}] "
            f"Aircraft monitored: {len(states.states)} | "
            f"Valid: {valid_aircraft} | "
            f"Max history: {maximum_history}/20 | "
            f"Ready: {ready_aircraft} | "
            f"Predictions: {total_predictions} | "
            f"Normal: {normal_predictions} | "
            f"Anomaly: {anomaly_predictions}"
        )

        time.sleep(COLLECTION_INTERVAL)

    except KeyboardInterrupt:

        print("\nAeroGuard stopped.")
        break

    except Exception as e:

        print("Error:", e)
        time.sleep(COLLECTION_INTERVAL)