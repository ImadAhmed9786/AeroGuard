
import pandas as pd
import numpy as np
import joblib

MODEL_FILE = "aeroguard_model_10.pkl"
DATA_FILE = "data/normal_flight_data.csv"
WINDOW_SIZE = 10

features = [
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
    "vertical_rate_max"
]

print("Loading 10-observation model...")
model = joblib.load(MODEL_FILE)

print("Loading normal OpenSky data...")
df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["icao24", "timestamp"])

windows = []

for aircraft_id, group in df.groupby("icao24"):
    group = group.dropna(
        subset=[
            "baro_altitude",
            "velocity",
            "heading",
            "vertical_rate"
        ]
    )

    for i in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = group.iloc[i:i + WINDOW_SIZE]

        windows.append({
            "altitude_mean": window["baro_altitude"].mean(),
            "altitude_std": window["baro_altitude"].std(ddof=0),
            "altitude_change": (
                window["baro_altitude"].iloc[-1]
                - window["baro_altitude"].iloc[0]
            ),
            "velocity_mean": window["velocity"].mean(),
            "velocity_std": window["velocity"].std(ddof=0),
            "velocity_change": (
                window["velocity"].iloc[-1]
                - window["velocity"].iloc[0]
            ),
            "heading_mean": window["heading"].mean(),
            "heading_std": window["heading"].std(ddof=0),
            "vertical_rate_mean": window["vertical_rate"].mean(),
            "vertical_rate_std": window["vertical_rate"].std(ddof=0),
            "vertical_rate_min": window["vertical_rate"].min(),
            "vertical_rate_max": window["vertical_rate"].max()
        })

print()
print("======================================")
print("NORMAL OPENSKY FALSE-POSITIVE TEST")
print("======================================")

if not windows:
    raise ValueError("No valid 10-observation windows found.")

test_df = pd.DataFrame(windows)

print("Normal windows:", len(test_df))

predictions = model.predict(test_df[features])

false_positives = int((predictions == 1).sum())
normal_predictions = int((predictions == 0).sum())
false_positive_rate = false_positives / len(predictions) * 100

print()
print("Predicted Normal    :", normal_predictions)
print("Predicted Emergency :", false_positives)
print(f"False-positive rate : {false_positive_rate:.2f}%")
print()
print("======================================")
print("TEST COMPLETE")
print("======================================")
