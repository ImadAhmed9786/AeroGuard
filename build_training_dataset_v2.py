import pandas as pd
import numpy as np

NORMAL_FILE = "data/normal_flight_data.csv"
EMERGENCY_FILE = "real_emergency_10sec.csv"
OUTPUT_FILE = "data/real_training_dataset_v2.csv"

WINDOW_SIZE = 20

print("Loading normal flight data...")
normal = pd.read_csv(NORMAL_FILE)

print("Loading emergency flight data...")
emergency = pd.read_csv(EMERGENCY_FILE)

# --------------------------------------------------
# Common cleaning
# --------------------------------------------------

print("Cleaning data...")

normal["timestamp"] = pd.to_datetime(normal["timestamp"], utc=True)
emergency["timestamp"] = pd.to_datetime(emergency["timestamp"], utc=True)

normal = normal.dropna(
    subset=["baro_altitude", "velocity", "heading", "vertical_rate"]
)

emergency = emergency.dropna(
    subset=["altitude", "groundspeed", "track", "vertical_rate"]
)

# --------------------------------------------------
# Feature creation
# --------------------------------------------------

def create_features(window):
    return {
        "altitude_mean": window["altitude"].mean(),
        "altitude_std": window["altitude"].std(),
        "altitude_change": window["altitude"].iloc[-1] - window["altitude"].iloc[0],

        "velocity_mean": window["velocity"].mean(),
        "velocity_std": window["velocity"].std(),
        "velocity_change": window["velocity"].iloc[-1] - window["velocity"].iloc[0],

        "heading_mean": window["heading"].mean(),
        "heading_std": window["heading"].std(),

        "vertical_rate_mean": window["vertical_rate"].mean(),
        "vertical_rate_std": window["vertical_rate"].std(),
        "vertical_rate_min": window["vertical_rate"].min(),
        "vertical_rate_max": window["vertical_rate"].max()
    }


# --------------------------------------------------
# NORMAL WINDOWS
# --------------------------------------------------

print("Creating normal-flight windows...")

normal = normal.sort_values(["icao24", "timestamp"])

normal_samples = []

for aircraft_id, group in normal.groupby("icao24"):

    group = group.sort_values("timestamp")

    for i in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_SIZE):

        window = group.iloc[i:i + WINDOW_SIZE]

        if len(window) < WINDOW_SIZE:
            continue

        temp = pd.DataFrame({
            "altitude": window["baro_altitude"].values,
            "velocity": window["velocity"].values,
            "heading": window["heading"].values,
            "vertical_rate": window["vertical_rate"].values
        })

        features = create_features(temp)

        features["group_id"] = f"normal_{aircraft_id}_{i}"
        features["label"] = 0

        normal_samples.append(features)

print("Normal windows:", len(normal_samples))


# --------------------------------------------------
# EMERGENCY WINDOWS
# --------------------------------------------------

print("Creating transition-based emergency windows...")

emergency = emergency.sort_values(["flight_id", "timestamp"])

emergency_samples = []

for flight_id, group in emergency.groupby("flight_id"):

    group = group.sort_values("timestamp").reset_index(drop=True)

    # Find first 7700 observation
    emergency_indices = group.index[
        group["squawk"].astype(str).str.zfill(4) == "7700"
    ]

    if len(emergency_indices) == 0:
        continue

    transition = emergency_indices[0]

    # ----------------------------------------------
    # Windows BEFORE emergency declaration
    # ----------------------------------------------

    for start in range(0, transition - WINDOW_SIZE + 1, WINDOW_SIZE):

        window = group.iloc[start:start + WINDOW_SIZE]

        if len(window) < WINDOW_SIZE:
            continue

        temp = pd.DataFrame({
            "altitude": window["altitude"].values,
            "velocity": window["groundspeed"].values,
            "heading": window["track"].values,
            "vertical_rate": window["vertical_rate"].values
        })

        features = create_features(temp)

        features["group_id"] = f"normal_from_{flight_id}_{start}"
        features["label"] = 0

        normal_samples.append(features)

    # ----------------------------------------------
    # Emergency windows
    # Start shortly before 7700 and continue after
    # ----------------------------------------------

    emergency_start = max(0, transition - WINDOW_SIZE)

    for start in range(
        emergency_start,
        len(group) - WINDOW_SIZE + 1,
        WINDOW_SIZE
    ):

        window = group.iloc[start:start + WINDOW_SIZE]

        if len(window) < WINDOW_SIZE:
            continue

        temp = pd.DataFrame({
            "altitude": window["altitude"].values,
            "velocity": window["groundspeed"].values,
            "heading": window["track"].values,
            "vertical_rate": window["vertical_rate"].values
        })

        features = create_features(temp)

        features["group_id"] = f"emergency_{flight_id}_{start}"
        features["label"] = 1

        emergency_samples.append(features)


print("Emergency windows:", len(emergency_samples))


# --------------------------------------------------
# Combine
# --------------------------------------------------

normal_df = pd.DataFrame(normal_samples)
emergency_df = pd.DataFrame(emergency_samples)

print()
print("======================================")
print("V2 DATASET BEFORE BALANCING")
print("======================================")

print("Normal:", len(normal_df))
print("Emergency:", len(emergency_df))

# Balance classes

sample_count = min(len(normal_df), len(emergency_df))

normal_df = normal_df.sample(
    n=sample_count,
    random_state=42
)

emergency_df = emergency_df.sample(
    n=sample_count,
    random_state=42
)

dataset = pd.concat(
    [normal_df, emergency_df],
    ignore_index=True
)

dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Remove accidental infinite values

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)

dataset = dataset.dropna()

# --------------------------------------------------
# Save
# --------------------------------------------------

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("======================================")
print("AEROGUARD V2 DATASET CREATED")
print("======================================")

print("Total samples:", len(dataset))
print("Normal samples:", (dataset["label"] == 0).sum())
print("Emergency samples:", (dataset["label"] == 1).sum())
print("Unique groups:", dataset["group_id"].nunique())

print()
print("Saved to:")
print(OUTPUT_FILE)