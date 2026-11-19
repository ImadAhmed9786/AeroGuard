import pandas as pd
import random
from datetime import datetime, timedelta

rows = []

start_time = datetime(2026, 9, 10, 16, 0, 0)

# Generate 500 flight observations
for i in range(500):

    timestamp = start_time + timedelta(seconds=i * 10)

    # Normal flight values
    altitude = random.uniform(28000, 38000)
    velocity = random.uniform(400, 500)
    heading = random.uniform(0, 360)
    vertical_rate = random.uniform(-5, 5)
    geo_altitude = altitude + random.uniform(-100, 100)
    latitude = random.uniform(15, 25)
    longitude = random.uniform(70, 85)

    # Create some abnormal descent examples
    if i % 20 == 0:
        vertical_rate = random.uniform(-800, -300)
        altitude -= random.uniform(500, 2000)
        label = 1
    else:
        label = 0

    rows.append({
        "timestamp": timestamp,
        "icao24": "AERO123",
        "callsign": "AI123",
        "longitude": longitude,
        "latitude": latitude,
        "baro_altitude": altitude,
        "velocity": velocity,
        "heading": heading,
        "vertical_rate": vertical_rate,
        "geo_altitude": geo_altitude,
        "on_ground": False,
        "label": label
    })

df = pd.DataFrame(rows)

df.to_csv("training_data.csv", index=False)

print("Training dataset created!")
print("Rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())