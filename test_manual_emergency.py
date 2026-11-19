import ast
from pathlib import Path

source_path = Path("aeroguard_single.py")
tree = ast.parse(source_path.read_text(encoding="utf-8"))

node = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "trigger_test_emergency"
)
module = ast.Module(body=[node], type_ignores=[])
ast.fix_missing_locations(module)

calls = {"transmit": 0, "save": 0}

def save_recording():
    calls["save"] += 1
    return None

def failing_transmitter(**kwargs):
    calls["transmit"] += 1
    raise OSError("Simulated transmitter failure")

env = {
    "test_emergency": True,
    "emergency_triggered": False,
    "save_previous_recording": save_recording,
    "activate_transmitter": failing_transmitter,
    "AIRCRAFT_ID": "TEST01",
    "CALLSIGN": "TEST",
    "print": lambda *args, **kwargs: None,
}
exec(compile(module, str(source_path), "exec"), env)

env["trigger_test_emergency"]()

assert env["emergency_triggered"] is True
assert env["test_emergency"] is False
assert calls["transmit"] == 1
print("PASS: Manual emergency remains latched after transmitter failure")
print("PASS: Transmitter exception is caught")
print("PASS: Transmitter called exactly once")

# Calling again must not trigger another transmission.
env["test_emergency"] = True
env["trigger_test_emergency"]()
assert calls["transmit"] == 1
print("PASS: No duplicate manual transmission")

print("\nAll manual emergency tests passed.")
