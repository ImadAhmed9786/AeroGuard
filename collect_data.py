from opensky_api import OpenSkyApi, TokenManager
from collections import deque
from datetime import datetime, timezone
import pandas as pd
import time
import os

# ==============================
# AEROGUARD SETTINGS
# ==============================

COLLECTION_INTERVAL = 10       # seconds
RECORDING_MINUTES = 45

# 45 minutes × 6 snapshots per minute
QUEUE_SIZE = (RECORDING_MINUTES * 60) // COLLECTION_INTERVAL

# ==============================
# TWO FIFO QUEUES
# ==============================

queue1 = deque(maxlen=QUEUE_SIZE)
queue2 = deque(maxlen=QUEUE_SIZE)

print("AEROGUARD started")
print(f"Queue 1: {QUEUE_SIZE} snapshots (45 minutes)")
print(f"Queue 2: {QUEUE_SIZE} snapshots (45 minutes backup)")
print("Press CTRL+C to stop.\n")

# ==============================
# OPEN SKY CONNECTION
# ==============================

token_manager = TokenManager.from_json_file("credentials.json")

with OpenSkyApi(token_manager=token_manager) as api:

    while True:

        try:
            print("Collecting aircraft data...")

            states = api.get_states()

            if states and states.states:

                rows = []

                # Time of this OpenSky snapshot
                observation_time = datetime.fromtimestamp(
                    states.time,
                    timezone.utc
                )

                # Store all aircraft in this snapshot
                for aircraft in states.states:

                    rows.append({
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

                # One complete snapshot = one queue node
                snapshot = pd.DataFrame(rows)

                # ==============================
                # MOVE OLD QUEUE 1 DATA → QUEUE 2
                # ==============================

                if len(queue1) == QUEUE_SIZE:

                    oldest_snapshot = queue1.popleft()

                    queue2.append(oldest_snapshot)

                # ==============================
                # ADD NEW SNAPSHOT TO QUEUE 1
                # ==============================

                queue1.append(snapshot)

                # ==============================
                # DISPLAY STATUS
                # ==============================

                print(f"Aircraft in snapshot: {len(snapshot)}")
                print(f"Queue 1: {len(queue1)}/{QUEUE_SIZE} snapshots")
                print(f"Queue 2: {len(queue2)}/{QUEUE_SIZE} snapshots")

                # ==============================
                # SAVE CURRENT RECORDING
                # ==============================

                os.makedirs("data", exist_ok=True)

                queue1_data = pd.concat(
                    list(queue1),
                    ignore_index=True
                )

                queue2_data = pd.concat(
                    list(queue2),
                    ignore_index=True
                ) if queue2 else pd.DataFrame()

                queue1_data.to_csv(
                    "data/queue1_recording.csv",
                    index=False
                )

                queue2_data.to_csv(
                    "data/queue2_backup.csv",
                    index=False
                )

                print("Queue 1 saved.")
                print("Queue 2 backup saved.\n")

            else:
                print("No aircraft data received.\n")

        except Exception as e:
            print(f"Collection error: {e}")
            print("Retrying...\n")

        time.sleep(COLLECTION_INTERVAL)