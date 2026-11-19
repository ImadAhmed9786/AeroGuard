
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

MODEL_FILE = "aeroguard_model_10.pkl"
DATA_FILE = "real_emergency_10sec.csv"
WINDOW_SIZE = 10

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max"
]

print("Loading saved AeroGuard model...")
model = joblib.load(MODEL_FILE)

print("Loading emergency flight data...")
df = pd.read_csv(DATA_FILE)
df = df.rename(columns={
    "altitude": "baro_altitude",
    "groundspeed": "velocity",
    "track": "heading"
})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.dropna(subset=[
    "flight_id", "timestamp", "baro_altitude",
    "velocity", "heading", "vertical_rate"
])
df = df.sort_values(["flight_id", "timestamp"])

windows = []

for flight_id, group in df.groupby("flight_id"):
    group = group.reset_index(drop=True)

    for start in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w = group.iloc[start:start + WINDOW_SIZE]

        altitude = w["baro_altitude"].to_numpy()
        velocity = w["velocity"].to_numpy()
        heading = w["heading"].to_numpy()
        vertical_rate = w["vertical_rate"].to_numpy()

        windows.append({
            "flight_id": flight_id,
            "altitude_mean": np.mean(altitude),
            "altitude_std": np.std(altitude, ddof=0),
            "altitude_change": altitude[-1] - altitude[0],
            "velocity_mean": np.mean(velocity),
            "velocity_std": np.std(velocity, ddof=0),
            "velocity_change": velocity[-1] - velocity[0],
            "heading_mean": np.mean(heading),
            "heading_std": np.std(heading, ddof=0),
            "vertical_rate_mean": np.mean(vertical_rate),
            "vertical_rate_std": np.std(vertical_rate, ddof=0),
            "vertical_rate_min": np.min(vertical_rate),
            "vertical_rate_max": np.max(vertical_rate)
        })

if not windows:
    raise ValueError("No valid emergency windows were created.")

test_df = pd.DataFrame(windows)
predictions = model.predict(test_df[FEATURES])

detected = int((predictions == 1).sum())
missed = int((predictions == 0).sum())
total = len(predictions)

print("\n======================================")
print("EMERGENCY-FLIGHT WINDOW TEST")
print("======================================")
print("Emergency windows:", total)
print("Flight groups:", test_df["flight_id"].nunique())
print("Predicted emergency:", detected)
print("Predicted normal:", missed)
print(f"Emergency-window detection rate: {detected / total * 100:.2f}%")
print(f"Missed-window rate: {missed / total * 100:.2f}%")
print("\nSaved model only; no retraining performed.")
print("======================================")
