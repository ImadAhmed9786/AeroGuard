from unittest.mock import Mock
from opensky_api import OpenSkyApi

print("AEROGUARD OPENSKY RECOVERY TEST")
print("=" * 45)

api = Mock(spec=OpenSkyApi)

# Simulate a failed request followed by a successful response.
api.get_states.side_effect = [
    OSError("Simulated connection reset"),
    None,
    Mock(states=[Mock(icao24="ab6fdd")]),
]

results = []

for attempt in range(1, 4):
    try:
        response = api.get_states(icao24=["ab6fdd"])
        if response is None or not response.states:
            results.append("NO DATA")
            print(f"Attempt {attempt}: no data, monitoring can continue")
        else:
            results.append("RECOVERED")
            print(f"Attempt {attempt}: response received successfully")
    except Exception as exc:
        results.append("ERROR")
        print(f"Attempt {attempt}: request error handled: {exc}")

assert results == ["ERROR", "NO DATA", "RECOVERED"]
assert api.get_states.call_count == 3

print("=" * 45)
print("PASS: Mocked request failure and recovery sequence")
print("Note: This tests the mock sequence, not the production monitoring loop.")
