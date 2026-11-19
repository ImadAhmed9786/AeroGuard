import ast
from pathlib import Path

SOURCE = Path("aeroguard_single.py")
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

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


def build_env(model, state, counts, transmitter):
    def save_recording():
        counts["save"] += 1
        return None

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
        "activate_transmitter": transmitter,
        "AIRCRAFT_ID": "TEST01",
        "CALLSIGN": "TEST",
        "print": lambda *args, **kwargs: None,
        "int": int,
        "float": float,
    }
    exec(compile(module, str(SOURCE), "exec"), env)
    return env


def main():
    model = FakeModel()
    state = {"time": 100}
    counts = {"save": 0, "transmit": 0}

    def successful_transmitter(**kwargs):
        counts["transmit"] += 1
        return True

    env = build_env(model, state, counts, successful_transmitter)
    run_ai = env["run_ai"]

    for _ in range(5):
        state["time"] += 10
        run_ai()

    check(env["emergency_triggered"], "Five qualifying windows trigger emergency")
    check(counts["transmit"] == 1, "Transmitter called exactly once")

    state["time"] += 10
    run_ai()
    check(counts["transmit"] == 1, "No repeat transmission after trigger")

    env["anomaly_count"] = 0
    env["emergency_triggered"] = False
    env["last_ai_window_end"] = None
    model.confidence = 0.849
    state["time"] += 10
    run_ai()

    check(env["anomaly_count"] == 0, "Low confidence resets streak")
    check(not env["emergency_triggered"], "Low confidence does not trigger")

    env["anomaly_count"] = 3
    model.prediction = 0
    model.confidence = 0.99
    state["time"] += 10
    run_ai()
    check(env["anomaly_count"] == 0, "Normal prediction resets streak")

    # Failure scenario: actual run_ai() with transmitter raising an exception.
    model.prediction = 1
    model.confidence = 0.99
    env["anomaly_count"] = 0
    env["emergency_triggered"] = False
    env["last_ai_window_end"] = None
    counts["transmit"] = 0

    def failing_transmitter(**kwargs):
        counts["transmit"] += 1
        raise OSError("Simulated transmitter failure")

    env["activate_transmitter"] = failing_transmitter

    for _ in range(5):
        state["time"] += 10
        run_ai()

    check(env["emergency_triggered"], "Emergency state remains latched after transmitter failure")
    check(counts["transmit"] == 1, "Failed transmission is not retried on later AI windows")

    state["time"] += 10
    run_ai()
    check(counts["transmit"] == 1, "Later AI windows do not repeat failed transmission")

    print("\nAll actual run_ai() tests passed.")


if __name__ == "__main__":
    main()
