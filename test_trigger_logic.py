# AeroGuard automatic-trigger logic test
# This tests the trigger rules independently of live OpenSky data.

ANOMALY_REQUIRED = 5
EMERGENCY_CONFIDENCE = 85.0


class TriggerTest:
    def __init__(self):
        self.anomaly_count = 0
        self.emergency_triggered = False
        self.trigger_count = 0

    def process(self, prediction, confidence):
        if self.emergency_triggered:
            return

        if prediction != 1 or confidence < EMERGENCY_CONFIDENCE:
            self.anomaly_count = 0
            return

        self.anomaly_count += 1

        if self.anomaly_count >= ANOMALY_REQUIRED:
            self.emergency_triggered = True
            self.trigger_count += 1


def check(condition, message):
    print(f"{'PASS' if condition else 'FAIL'}: {message}")


# Test 1: Five consecutive qualifying windows
test = TriggerTest()

for _ in range(5):
    test.process(1, 90.0)

check(test.emergency_triggered, "Five qualifying windows trigger emergency")
check(test.trigger_count == 1, "Emergency triggers exactly once")

# Test 2: A confidence score below 85% resets the streak
test = TriggerTest()

for _ in range(4):
    test.process(1, 90.0)

test.process(1, 84.9)

check(test.anomaly_count == 0, "Low confidence resets the streak")
check(not test.emergency_triggered, "Four qualifying windows do not trigger")

# Test 3: A normal prediction resets the streak
test = TriggerTest()

for _ in range(4):
    test.process(1, 90.0)

test.process(0, 99.0)

check(test.anomaly_count == 0, "Normal prediction resets the streak")

# Test 4: A triggered emergency cannot trigger twice
test = TriggerTest()

for _ in range(10):
    test.process(1, 95.0)

check(test.trigger_count == 1, "Emergency triggers only once")

print("\nTrigger-rule tests completed.")