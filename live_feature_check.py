import pandas as pd
import numpy as np
import joblib

# ==============================
# FILES
# ==============================

MODEL_FILE = "aeroguard_model.pkl"
NORMAL_FILE = "data/normal_flight_data.csv"
TRAIN_FILE = "data/real_training_dataset.csv"

# ==============================
# LOAD MODEL + DATA
# ==============================

model = joblib.load(MODEL_FILE)

normal = pd.read_csv(NORMAL_FILE)
train = pd.read_csv(TRAIN_FILE)

normal["timestamp"] = pd.to_datetime(
    normal["timestamp"],
    utc=True
)

print("\n==============================")
print("AEROGUARD LIVE FEATURE CHECK")
print("==============================")

print("\nNormal OpenSky rows:", len(normal))
print("Training rows:", len(train))


# ============================================================
# TIMESTAMP SPACING CHECK
# ============================================================

print("\n==============================================")
print("TIMESTAMP SPACING CHECK")
print("==============================================")

normal = normal.sort_values(
    ["icao24", "timestamp"]
)

normal["time_diff"] = (
    normal.groupby("icao24")["timestamp"]
    .diff()
    .dt.total_seconds()
)

time_diff = normal["time_diff"].dropna()

print("\nAll consecutive observations:")

print(
    "Minimum gap :",
    f"{time_diff.min():.2f} seconds"
)

print(
    "Maximum gap :",
    f"{time_diff.max():.2f} seconds"
)

print(
    "Mean gap    :",
    f"{time_diff.mean():.2f} seconds"
)

print(
    "Median gap  :",
    f"{time_diff.median():.2f} seconds"
)
print("\nGap distribution:")

print(
    time_diff.describe().to_string()
)

print("\nObservations close to 10 seconds:")

for tolerance in [1, 2, 3, 5]:

    count = (
        (time_diff >= 10 - tolerance)
        &
        (time_diff <= 10 + tolerance)
    ).sum()

    percentage = (
        count / len(time_diff) * 100
    )

    print(
        f"Within ±{tolerance}s : "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# FEATURE NAMES
# ============================================================

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


# ============================================================
# BUILD NORMAL WINDOWS
# ============================================================

normal_windows = []

window_gaps = []

for aircraft_id, group in normal.groupby("icao24"):

    group = group.sort_values("timestamp")

    values = group[
        [
            "baro_altitude",
            "velocity",
            "heading",
            "vertical_rate"
        ]
    ].dropna()

    timestamps = group.loc[
        values.index,
        "timestamp"
    ]

    if len(values) < 20:
        continue

    for i in range(
        0,
        len(values) - 19,
        20
    ):

        window = values.iloc[
            i:i + 20
        ]

        window_times = timestamps.iloc[
            i:i + 20
        ]

        if len(window) < 20:
            continue

        # Time span of this 20-observation window
        span = (
            window_times.iloc[-1]
            - window_times.iloc[0]
        ).total_seconds()

        window_gaps.append(span)

        altitude = window[
            "baro_altitude"
        ].to_numpy()

        velocity = window[
            "velocity"
        ].to_numpy()

        heading = window[
            "heading"
        ].to_numpy()

        vertical_rate = window[
            "vertical_rate"
        ].to_numpy()

        row = [

            np.mean(altitude),
            np.std(
                altitude,
                ddof=1
            ),
            altitude[-1] - altitude[0],

            np.mean(velocity),
            np.std(
                velocity,
                ddof=1
            ),
            velocity[-1] - velocity[0],

            np.mean(heading),
            np.std(
                heading,
                ddof=1
            ),

            np.mean(vertical_rate),
            np.std(
                vertical_rate,
                ddof=1
            ),
            np.min(vertical_rate),
            np.max(vertical_rate)
        ]

        normal_windows.append(row)


live = pd.DataFrame(
    normal_windows,
    columns=features
)

print("\nLive normal windows:", len(live))


# ============================================================
# WINDOW TIME-SPAN CHECK
# ============================================================

window_gaps = np.array(
    window_gaps,
    dtype=float
)

print("\n==============================================")
print("20-OBSERVATION WINDOW SPAN")
print("==============================================")

print(
    "Minimum span :",
    f"{window_gaps.min():.2f} seconds"
)

print(
    "Maximum span :",
    f"{window_gaps.max():.2f} seconds"
)

print(
    "Mean span    :",
    f"{window_gaps.mean():.2f} seconds"
)

print(
    "Median span  :",
    f"{np.median(window_gaps):.2f} seconds"
)

print(
    "\nExpected span for 20 observations "
    "at 10-second intervals: ~190 seconds"
)


# ============================================================
# FEATURE RANGE COMPARISON
# ============================================================

print("\n==============================================")
print("FEATURE RANGE COMPARISON")
print("==============================================")

for feature in features:

    train_min = train[feature].min()
    train_max = train[feature].max()
    train_mean = train[feature].mean()

    live_min = live[feature].min()
    live_max = live[feature].max()
    live_mean = live[feature].mean()

    print(f"\n{feature}")

    print(
        f"  TRAIN : min={train_min:.3f} "
        f"max={train_max:.3f} "
        f"mean={train_mean:.3f}"
    )

    print(
        f"  LIVE  : min={live_min:.3f} "
        f"max={live_max:.3f} "
        f"mean={live_mean:.3f}"
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

X_live = live[features]

predictions = model.predict(
    X_live
)

probabilities = model.predict_proba(
    X_live
)

anomalies = np.sum(
    predictions == 1
)

normals = np.sum(
    predictions == 0
)

print("\n==============================================")
print("MODEL RESULT")
print("==============================================")

print(
    "Total windows :",
    len(predictions)
)

print(
    "Normal        :",
    normals
)

print(
    "Anomaly       :",
    anomalies
)

if len(predictions) > 0:

    fpr = (
        anomalies /
        len(predictions)
    ) * 100

    print(
        "False positive rate:",
        f"{fpr:.2f}%"
    )


# ============================================================
# ANOMALY CONFIDENCE
# ============================================================

anomaly_probabilities = (
    probabilities[
        predictions == 1,
        1
    ] * 100
)

print("\n==============================================")
print("ANOMALY CONFIDENCE")
print("==============================================")

if len(anomaly_probabilities) > 0:

    print(
        "Anomalies found:",
        len(anomaly_probabilities)
    )

    print(
        "Minimum confidence:",
        f"{anomaly_probabilities.min():.2f}%"
    )

    print(
        "Maximum confidence:",
        f"{anomaly_probabilities.max():.2f}%"
    )

    print(
        "Average confidence:",
        f"{anomaly_probabilities.mean():.2f}%"
    )

else:

    print(
        "No anomalies detected."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("MODEL FEATURE IMPORTANCE")
print("==============================================")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(
    "importance",
    ascending=False
)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("CHECK COMPLETE")
print("==============================================")