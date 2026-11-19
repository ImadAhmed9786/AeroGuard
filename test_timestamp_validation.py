from datetime import datetime, timezone

source = open("response_processing_section.txt", encoding="utf-8").read()

assert "age > 60 or age < -10" in source
assert "observation_time <= last_observation_timestamp" in source

now = datetime.now(timezone.utc).timestamp()

# Mirror the exact production conditions.
def accepted(observation_time, last_timestamp, current_time):
    age = current_time - observation_time
    if age > 60 or age < -10:
        return False
    if last_timestamp is not None and observation_time <= last_timestamp:
        return False
    return True

assert accepted(now, None, now)
print("PASS: Fresh observation accepted")

assert not accepted(now - 120, None, now)
print("PASS: Stale observation rejected")

assert not accepted(now, now, now)
print("PASS: Duplicate observation rejected")

assert not accepted(now + 20, None, now)
print("PASS: Future-dated observation rejected")

print("\nVALIDATION TESTS PASSED")
print("Note: The production conditions were verified, but this helper is not the full polling loop.")
