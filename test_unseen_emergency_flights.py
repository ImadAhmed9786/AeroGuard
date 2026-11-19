import pandas as pd
import numpy as np
import joblib

MODEL_FILE = "aeroguard_model_10.pkl"
DATA_FILE = "real_emergency_10sec_clean_v3.csv"
WINDOW_SIZE = 10

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max"
]

CANDIDATES = [
    "AEE4258_20191007", "AMX057_20200128", "BOX495_20190907",
    "FDX1429_20190808", "IBK3502_20190915", "JBU570_20181021",
    "MSC610_20180529", "OAE351_20190423", "SWA3794_20191223"
]

print("Loading saved model...")
model = joblib.load(MODEL_FILE)

print("Loading cleaned emergency telemetry...")
df = pd.read_csv(DATA_FILE)
df["flight_id"] = df["flight_id"].astype(str).str.strip()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df[df["flight_id"].isin(CANDIDATES)].copy()

df = df.rename(columns={
    "altitude": "baro_altitude",
    "groundspeed": "velocity",
    "track": "heading"
})

required = [
    "flight_id", "timestamp", "baro_altitude",
    "velocity", "heading", "vertical_rate"
]
df = df.dropna(subset=required)
df = df.sort_values(["flight_id", "timestamp"])

windows = []

for flight_id, group in df.groupby("flight_id"):
    group = group.reset_index(drop=True)

    for start in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w = group.iloc[start:start + WINDOW_SIZE]

        altitude = w["baro_altitude"].to_numpy(dtype=float)
        velocity = w["velocity"].to_numpy(dtype=float)
        heading = w["heading"].to_numpy(dtype=float)
        vertical_rate = w["vertical_rate"].to_numpy(dtype=float)

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
    raise ValueError("No complete telemetry windows found.")

test_df = pd.DataFrame(windows)
test_df["prediction"] = model.predict(test_df[FEATURES])

print("\nUNSEEN EMERGENCY FLIGHT TEST")
print("=" * 55)

total_detected = 0
total_windows = len(test_df)

for flight_id, group in test_df.groupby("flight_id"):
    detected = int((group["prediction"] == 1).sum())
    total = len(group)
    total_detected += detected
    print(
        f"{flight_id}: {detected}/{total} windows flagged "
        f"({detected / total * 100:.1f}%)"
    )

print("=" * 55)
print(f"Total windows: {total_windows}")
print(f"Emergency-class predictions: {total_detected}")
print(f"Window detection rate: {total_detected / total_windows * 100:.2f}%")
print("\nTEST COMPLETE")
print("Note: This measures model predictions, not verified real-time alert performance.")
