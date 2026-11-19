
import time
import random
import requests
from datetime import datetime, timezone

from opensky_api import TokenManager


# Configuration
CREDENTIALS_FILE = "credentials.json"
AIRCRAFT_ICAO24 = "39de4d"

API_URL = "https://opensky-network.org/api/states/all"

# Do not poll more frequently than this in this test.
POLL_INTERVAL_SECONDS = 60

# Prevent an endless diagnostic run.
MAX_REQUESTS = 10

# Honor the server's Retry-After response.
MAX_RETRY_WAIT_SECONDS = 24 * 60 * 60


def get_retry_after(response):
    """Read Retry-After safely; fall back to exponential backoff."""
    value = response.headers.get("Retry-After")

    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass

    return None


def main():
    print("AEROGUARD: RATE-LIMIT-SAFE OPENSKY TEST")
    print(f"Aircraft: {AIRCRAFT_ICAO24}")
    print("This test does not modify your model or recording files.")

    try:
        token_manager = TokenManager.from_json_file(
            CREDENTIALS_FILE
        )
        print("Credentials loaded.")
    except Exception as exc:
        print(f"Could not load credentials: {exc}")
        return

    session = requests.Session()
    request_count = 0
    consecutive_errors = 0

    while request_count < MAX_REQUESTS:
        request_count += 1

        print(
            f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"Request {request_count}/{MAX_REQUESTS}"
        )

        try:
            response = session.get(
                API_URL,
                params={"icao24": AIRCRAFT_ICAO24},
                headers=token_manager.auth_headers(),
                timeout=(5, 20),
            )

            print(f"HTTP status: {response.status_code}")

            if response.status_code == 429:
                retry_after = get_retry_after(response)

                if retry_after is None:
                    retry_after = min(
                        300 * (2 ** consecutive_errors),
                        MAX_RETRY_WAIT_SECONDS,
                    )

                print("OpenSky rate limit reached.")
                print(f"Server retry delay: {retry_after} seconds.")

                if retry_after > MAX_RETRY_WAIT_SECONDS:
                    print(
                        "Retry delay exceeds this test's limit. "
                        "Stopping without sending more requests."
                    )
                    break

                print("Waiting for the server's retry period...")
                time.sleep(retry_after + random.uniform(1, 5))
                consecutive_errors += 1
                continue

            if response.status_code in (401, 403):
                print(
                    "Authentication or access denied. "
                    "Check your OpenSky account and credentials."
                )
                break

            response.raise_for_status()
            data = response.json()
            states = data.get("states") or []

            if not states:
                print(
                    "No state returned for this aircraft. "
                    "It may not currently be available in OpenSky."
                )
            else:
                state = states[0]
                last_contact = state[4]
                latitude = state[6]
                longitude = state[5]
                on_ground = state[8]

                print(f"ICAO24: {state[0]}")
                print(f"Callsign: {(state[1] or '').strip()}")
                print(f"On ground: {on_ground}")
                print(f"Latitude: {latitude}")
                print(f"Longitude: {longitude}")

                if last_contact is not None:
                    age = time.time() - last_contact
                    print(f"Last contact age: {age:.1f} seconds")

            consecutive_errors = 0

        except requests.exceptions.Timeout:
            consecutive_errors += 1
            print("Request timed out. No immediate retry.")

        except requests.exceptions.RequestException as exc:
            consecutive_errors += 1
            print(f"Network/API error: {exc}")

        except ValueError as exc:
            consecutive_errors += 1
            print(f"Invalid JSON response: {exc}")

        if request_count < MAX_REQUESTS:
            delay = min(
                POLL_INTERVAL_SECONDS * (2 ** consecutive_errors),
                15 * 60,
            )
            print(f"Next request in {delay} seconds.")
            time.sleep(delay)

    session.close()
    print("\nDiagnostic finished.")


if __name__ == "__main__":
    main()
