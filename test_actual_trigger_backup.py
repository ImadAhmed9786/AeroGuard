import ast
from pathlib import Path

SOURCE = Path("aeroguard_single.py")
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

# Extract the real run_ai() function without running the live monitor.
node = next(
    item for item in tree.body
    if isinstance(item, ast.FunctionDef) and item.name == "run_ai"
)

module = ast.Module(body=[node], type_ignores=[])
ast.fix_missing_locations(module)


class FakeModel:
    classes_ = [0, 1]

    def __init__(self):
        self.prediction = 1
        self.confidence = 0.90

    def predict(self, features):
        return [self.prediction]

    def predict_proba(self, features):
        return [[1 - self.confidence, self.confidence]]


class FakeWindow:
    def __init__(self, timestamp):
        self.items = [
            {"timestamp_unix": timestamp - 90 + i * 10}
            for i in range(10)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def check(condition, message):
    if not condition:
        raise AssertionError("FAIL: " + message)
    print("PASS: " + message)


def main():
    model = FakeModel()
    state = {"time": 100}
    counts = {"save": 0, "transmit": 0}

    def save_recording():
        counts["save"] += 1
        return None

    def transmit(**kwargs):
        counts["transmit"] += 1
        return True

    env = {
        "ANOMALY_REQUIRED": 5,
        "EMERGENCY_CONFIDENCE": 85.0,
        "AI_WINDOW": 10,
        "anomaly_count": 0,
        "emergency_triggered": False,
        "last_ai_window_end": None,
        "model": model,
        "get_ai_window": lambda: FakeWindow(state["time"]),
        "create_features": lambda window: object(),
        "save_previous_recording": save_recording,
        "activate_transmitter": transmit,
        "AIRCRAFT_ID": "TEST01",
        "CALLSIGN": "TEST",
        "print": lambda *args, **kwargs: None,
        "int": int,
        "float": float,
    }

    exec(compile(module, str(SOURCE), "exec"), env)
    run_ai = env["run_ai"]

    # Five distinct qualifying windows should trigger once.
    for _ in range(5):
        state["time"] += 10
        run_ai()

    check(
        env["emergency_triggered"] is True,
        "Five qualifying windows trigger the emergency"
    )
    check(
        counts["transmit"] == 1,
        "Emergency transmission handler called once"
    )

    # Further qualifying windows must not transmit again.
    state["time"] += 10
    run_ai()

    check(
        counts["transmit"] == 1,
        "Subsequent windows do not trigger another transmission"
    )

    # Reset state and test confidence below 85%.
    env["anomaly_count"] = 0
    env["emergency_triggered"] = False
    env["last_ai_window_end"] = None
    model.confidence = 0.849
    state["time"] += 10
    run_ai()

    check(
        env["anomaly_count"] == 0,
        "Confidence below 85% resets the streak"
    )
    check(
        env["emergency_triggered"] is False,
        "Low confidence does not trigger an emergency"
    )
    check(
        counts["transmit"] == 1,
        "Low confidence causes no additional transmission"
    )

    # A normal prediction must also reset the streak.
    env["anomaly_count"] = 3
    model.prediction = 0
    model.confidence = 0.99
    state["time"] += 10
    run_ai()

    check(
        env["anomaly_count"] == 0,
        "Normal prediction resets the streak"
    )

    print("\nAll actual run_ai() tests passed.")


if __name__ == "__main__":
    main()

