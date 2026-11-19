from unittest.mock import Mock
from types import SimpleNamespace

# Mock the API only. No live OpenSky requests.
api = Mock()
api.get_states.side_effect = [
    ConnectionError("Simulated connection reset"),
    SimpleNamespace(states=[]),
    SimpleNamespace(states=[SimpleNamespace(icao24="test123")]),
]

attempts = 0
results = []

for _ in range(3):  # Bounded: cannot run forever
    attempts += 1
    try:
        response = api.get_states(icao24=["test123"])
    except Exception as exc:
        results.append(("error", str(exc)))
        continue

    if response is None or not response.states:
        results.append(("empty", None))
        continue

    results.append(("success", response.states[0].icao24))

assert attempts == 3
assert [item[0] for item in results] == ["error", "empty", "success"]
assert results[-1][1] == "test123"

print("PASS: First polling attempt handled a connection error.")
print("PASS: Second polling attempt handled an empty response.")
print("PASS: Third polling attempt received a successful response.")
print("PASS: Three bounded attempts completed.")
print("\nNote: This verifies the recovery sequence using a mock API, not the full production loop.")
