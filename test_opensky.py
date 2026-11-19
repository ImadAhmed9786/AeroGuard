
import time
import requests
from datetime import datetime, timezone
from opensky_api import TokenManager

AIRCRAFT_ID = "39de4d"
URL = "https://opensky-network.org/api/states/all"


def main():
    print("AEROGUARD: OPENSKY CONNECTION TEST")
    print("Aircraft:", AIRCRAFT_ID)

    try:
        token_manager = TokenManager.from_json_file(
            "credentials.json"
        )
        print("Credentials file loaded.")
    except Exception as exc:
        print("CREDENTIALS ERROR:", repr(exc))
        return

    with requests.Session() as session:
        for attempt in range(1, 4):
            print(f"\n--- Request {attempt}/3 ---")

            try:
                response = session.get(
                    URL,
                    params={"icao24": AIRCRAFT_ID},
                    headers=token_manager.auth_headers(),
                    timeout=(5, 20),
                )

                print("HTTP status:", response.status_code)
                print(
                    "Rate-limit remaining:",
                    response.headers.get(
                        "X-Rate-Limit-Remaining", "Not provided"
                    ),
                )
                print(
                    "Retry-after:",
                    response.headers.get(
                        "X-Rate-Limit-Retry-After-Seconds",
                        "Not provided",
                    ),
                )

                if response.status_code != 200:
                    print("Response:", response.text[:500])

                    if response.status_code in (401, 403, 429):
                        print(
                            "Authentication, permission, or "
                            "rate-limit issue indicated."
                        )
                        break

                else:
                    data = response.json()
                    states = data.get("states") or []
                    print("States returned:", len(states))

                    if not states:
                        print(
                            "Request succeeded, but no state vectors "
                            "were returned for this aircraft."
                        )
                    else:
                        for state in states[:3]:
                            print("ICAO24:", state[0])
                            print(
                                "Callsign:",
                                (state[1] or "").strip(),
                            )
                            print("Position time:", state[3])
                            print("Last contact:", state[4])
                            print("Altitude:", state[7])
                            print("On ground:", state[8])

                            if state[3]:
                                age = (
                                    datetime.now(timezone.utc).timestamp()
                                    - state[3]
                                )
                                print(f"Position age: {age:.1f} seconds")

            except requests.RequestException as exc:
                print("NETWORK ERROR:", repr(exc))
            except Exception as exc:
                print("OTHER ERROR:", repr(exc))
                break

            if attempt < 3:
                print("Waiting 10 seconds...")
                time.sleep(10)

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
