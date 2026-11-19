
import pandas as pd
from pathlib import Path

INPUT_FILE = "squawk7700_trajectories.parquet.gz"
OUTPUT_FILE = "real_emergency_10sec_clean_v3.csv"

# Conservative screening settings for this research prototype.
MAX_ALTITUDE_FT = 60000
MAX_GROUNDSPEED_KNOTS = 600
MAX_VERTICAL_RATE_FPM = 10000
MAX_ALTITUDE_RATE_MPS = 30
TIME_WINDOW_SECONDS = 60

print("Loading original dataset...")
df = pd.read_parquet(INPUT_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["flight_id", "timestamp"]).copy()

print(f"Original rows: {len(df):,}")

# Basic range checks on source units.
valid = (
    df["altitude"].between(0, MAX_ALTITUDE_FT)
    & df["groundspeed"].between(0, MAX_GROUNDSPEED_KNOTS)
    & df["vertical_rate"].abs().le(MAX_VERTICAL_RATE_FPM)
    & df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
)

required = [
    "flight_id", "icao24", "timestamp", "latitude", "longitude",
    "altitude", "groundspeed", "vertical_rate", "track"
]

df = df.loc[valid].dropna(subset=required).copy()
df = df.sort_values(["flight_id", "timestamp"])

print(f"Rows after basic checks: {len(df):,}")

# Iteratively exclude short segments around extreme altitude jumps.
# This is a heuristic for a candidate dataset, not a certified rule.
iteration = 0

while True:
    iteration += 1

    group = df.groupby("flight_id", sort=False)
    previous_altitude = group["altitude"].shift()
    previous_time = group["timestamp"].shift()

    seconds = (df["timestamp"] - previous_time).dt.total_seconds()
    altitude_rate_mps = (
        (df["altitude"] - previous_altitude).abs()
        * 0.3048
        / seconds
    )

    jump = (
        seconds.gt(0)
        & seconds.le(30)
        & altitude_rate_mps.gt(MAX_ALTITUDE_RATE_MPS)
    )

    jump_times = df.loc[jump, ["flight_id", "timestamp"]]

    if jump_times.empty:
        print(f"No further extreme jumps detected after {iteration - 1} passes.")
        break

    print(
        f"Pass {iteration}: {len(jump_times):,} extreme transitions; "
        "excluding nearby samples."
    )

    # Exclude samples within 60 seconds of each detected jump
    # in the same flight, to remove suspicious plateaus as well.
    remove = pd.Series(False, index=df.index)

    for flight_id, part in jump_times.groupby("flight_id"):
        flight_rows = df.loc[df["flight_id"] == flight_id, "timestamp"]

        for event_time in part["timestamp"]:
            nearby = (
                flight_rows.sub(event_time).abs()
                <= pd.Timedelta(seconds=TIME_WINDOW_SECONDS)
            )
            remove.loc[nearby.index[nearby]] = True

    before = len(df)
    df = df.loc[~remove].copy()
    print(f"Samples excluded this pass: {before - len(df):,}")

    if df.empty:
        raise RuntimeError("Cleaning removed all data; review the thresholds.")

# Convert source units to metric units.
df["altitude"] *= 0.3048
df["groundspeed"] *= 0.514444
df["vertical_rate"] *= 0.00508

# Keep one actual source observation per 10-second bin.
df["time_bin"] = df["timestamp"].dt.floor("10s")
df = df.drop_duplicates(["flight_id", "time_bin"], keep="first")
df = df.drop(columns=["time_bin"])
df = df.sort_values(["flight_id", "timestamp"])

df.to_csv(OUTPUT_FILE, index=False)

print("\nCandidate dataset created.")
print(f"Output rows: {len(df):,}")
print(f"Flights retained: {df['flight_id'].nunique():,}")
print(f"Saved to: {Path(OUTPUT_FILE).resolve()}")
print("\nThis file is separate from the previous candidates and live model.")
print("Validate before training.")
