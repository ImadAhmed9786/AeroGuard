from opensky_api import OpenSkyApi, TokenManager
from datetime import datetime, timezone
import pandas as pd
import time
import os

COLLECTION_INTERVAL = 10
COLLECTION_TIME = 5 * 60  # 5 minutes

OUTPUT_FILE = "data/normal_flight_data.csv"

token_manager = TokenManager.from_json_file("credentials.json")

all_rows = []
start_time = time.time()

with OpenSkyApi(token_manager=token_manager) as api:

    print("Starting normal flight data collection...")

    while time.time() - start_time < COLLECTION_TIME:

        try:
            states = api.get_states()

            if states and states.states:

                observation_time = datetime.fromtimestamp(
                    states.time,
                    timezone.utc
                )

                for aircraft in states.states:

                    # Ignore aircraft on the ground
                    if aircraft.on_ground:
                        continue

                    all_rows.append({
                        "timestamp": observation_time,
                        "icao24": aircraft.icao24,
                        "callsign": aircraft.callsign,
                        "longitude": aircraft.longitude,
                        "latitude": aircraft.latitude,
                        "baro_altitude": aircraft.baro_altitude,
                        "velocity": aircraft.velocity,
                        "heading": aircraft.true_track,
                        "vertical_rate": aircraft.vertical_rate,
                        "geo_altitude": aircraft.geo_altitude,
                        "on_ground": aircraft.on_ground,
                        "squawk": aircraft.squawk
                    })

                print(
                    f"{observation_time} | "
                    f"Aircraft collected: {len(states.states)} | "
                    f"Total rows: {len(all_rows)}"
                )

        except Exception as e:
            print("Error:", e)

        time.sleep(COLLECTION_INTERVAL)

df = pd.DataFrame(all_rows)

os.makedirs("data", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nCollection finished!")
print("Total rows:", len(df))
print("Unique aircraft:", df["icao24"].nunique())
print("Saved to:", OUTPUT_FILE)