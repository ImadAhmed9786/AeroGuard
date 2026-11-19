from opensky_api import OpenSkyApi, TokenManager
import time


# Load OpenSky credentials
token_manager = TokenManager.from_json_file("credentials.json")


print("AEROGUARD LIVE SYSTEM STARTED")
print("Connecting to OpenSky...\n")


with OpenSkyApi(token_manager=token_manager) as api:

    while True:

        try:
            states = api.get_states()

            if states and states.states:

                print("Aircraft detected:", len(states.states))

                # Show first 5 aircraft
                for aircraft in states.states[:5]:

                    speed_kmh = (
                        aircraft.velocity * 3.6
                        if aircraft.velocity is not None
                        else None
                    )

                    print(
                        "ICAO:", aircraft.icao24,
                        "| Callsign:", aircraft.callsign,
                        "| Altitude:",
                        f"{aircraft.baro_altitude:.2f} m"
                        if aircraft.baro_altitude is not None
                        else "N/A",
                        "| Speed:",
                        f"{speed_kmh:.2f} km/h"
                        if speed_kmh is not None
                        else "N/A",
                        "| Heading:",
                        f"{aircraft.true_track:.2f}°"
                        if aircraft.true_track is not None
                        else "N/A",
                        "| Vertical rate:",
                        f"{aircraft.vertical_rate:.2f} m/s"
                        if aircraft.vertical_rate is not None
                        else "N/A"
                    )

                print("\n" + "-" * 60)
                print("Waiting 10 seconds...\n")

            else:
                print("No aircraft data received.")

            time.sleep(10)

        except Exception as e:
            print("Error:", e)
            print("Retrying in 10 seconds...\n")

            time.sleep(10)