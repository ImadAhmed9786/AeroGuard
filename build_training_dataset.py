import pandas as pd
import numpy as np


# settings

NORMAL_FILE = "data/normal_flight_data.csv"
EMERGENCY_FILE = "real_emergency_10sec.csv"
OUTPUT_FILE = "data/real_training_dataset_10.csv"

WINDOW_SIZE = 10


# load data

print("Loading normal flight data...")
normal = pd.read_csv(NORMAL_FILE)

print("Loading emergency flight data...")
emergency = pd.read_csv(EMERGENCY_FILE)


# standardise columns

emergency = emergency.rename(columns={
    "altitude": "baro_altitude",
    "groundspeed": "velocity",
    "track": "heading"
})


# select features

normal = normal[
    [
        "timestamp",
        "icao24",
        "baro_altitude",
        "velocity",
        "heading",
        "vertical_rate"
    ]
]

emergency = emergency[
    [
        "timestamp",
        "flight_id",
        "icao24",
        "baro_altitude",
        "velocity",
        "heading",
        "vertical_rate"
    ]
]


# clean data

print("Cleaning data...")

normal["timestamp"] = pd.to_datetime(
    normal["timestamp"],
    errors="coerce"
)

emergency["timestamp"] = pd.to_datetime(
    emergency["timestamp"],
    errors="coerce"
)

required_columns = [
    "timestamp",
    "baro_altitude",
    "velocity",
    "heading",
    "vertical_rate"
]

normal = normal.dropna(subset=required_columns)
emergency = emergency.dropna(subset=required_columns)


# sort data

normal = normal.sort_values(
    ["icao24", "timestamp"]
)

emergency = emergency.sort_values(
    ["flight_id", "timestamp"]
)


# create windows

def create_windows(df, group_column, label):

    samples = []

    for group_id, group in df.groupby(group_column):

        group = group.reset_index(drop=True)

        if len(group) < WINDOW_SIZE:
            continue

        for start in range(
            0,
            len(group) - WINDOW_SIZE + 1,
            WINDOW_SIZE
        ):

            window = group.iloc[
                start:start + WINDOW_SIZE
            ]

            if len(window) < WINDOW_SIZE:
                continue

            altitude = window["baro_altitude"].values
            velocity = window["velocity"].values
            heading = window["heading"].values
            vertical_rate = window["vertical_rate"].values

            sample = {
                "group_id": group_id,

                "altitude_mean": np.mean(altitude),
                "altitude_std": np.std(altitude),
                "altitude_change": altitude[-1] - altitude[0],

                "velocity_mean": np.mean(velocity),
                "velocity_std": np.std(velocity),
                "velocity_change": velocity[-1] - velocity[0],

                "heading_mean": np.mean(heading),
                "heading_std": np.std(heading),

                "vertical_rate_mean": np.mean(vertical_rate),
                "vertical_rate_std": np.std(vertical_rate),
                "vertical_rate_min": np.min(vertical_rate),
                "vertical_rate_max": np.max(vertical_rate),

                "label": label
            }

            samples.append(sample)

    return pd.DataFrame(samples)


# normal windows

print("Creating normal-flight windows...")

normal_windows = create_windows(
    normal,
    "icao24",
    0
)

print("Normal windows:", len(normal_windows))


# emergency windows

print("Creating emergency-flight windows...")

emergency_windows = create_windows(
    emergency,
    "flight_id",
    1
)

print("Emergency windows:", len(emergency_windows))


# balance data

print("Balancing dataset...")

sample_count = min(
    len(normal_windows),
    len(emergency_windows)
)

normal_windows = normal_windows.sample(
    n=sample_count,
    random_state=42
)

emergency_windows = emergency_windows.sample(
    n=sample_count,
    random_state=42
)


# combine

dataset = pd.concat(
    [
        normal_windows,
        emergency_windows
    ],
    ignore_index=True
)

dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# result

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n======================================")
print("REAL TRAINING DATASET CREATED")
print("======================================")

print("Total samples:", len(dataset))
print("Normal samples:", (dataset["label"] == 0).sum())
print("Emergency samples:", (dataset["label"] == 1).sum())
print("Unique groups:", dataset["group_id"].nunique())

print("\nColumns:")
print(dataset.columns.tolist())

print("\nSaved to:")
print(OUTPUT_FILE)