from pathlib import Path

p = Path("aeroguard_single.py")
s = p.read_text(encoding="utf-8")

old = '''        observation_time = float(state.time_position)

        if (
            last_observation_timestamp is not None
            and observation_time <= last_observation_timestamp
        ):
'''

new = '''        observation_time = float(state.time_position)
        current_time = datetime.now(timezone.utc).timestamp()

        # Reject observations that are too old or implausibly future-dated.
        observation_age = current_time - observation_time
        if observation_age > 60 or observation_age < -10:
            print(
                f"Stale or future-dated observation skipped "
                f"(age: {observation_age:.1f} seconds)."
            )
            continue

        if (
            last_observation_timestamp is not None
            and observation_time <= last_observation_timestamp
        ):
'''

if s.count(old) != 1:
    raise SystemExit(
        "Patch stopped: expected timestamp block not found exactly once. "
        "No changes made."
    )

updated = s.replace(old, new, 1)
compile(updated, str(p), "exec")
p.write_text(updated, encoding="utf-8")
print("PATCH APPLIED: stale/future timestamp guard added.")
