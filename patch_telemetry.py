from pathlib import Path

path = Path("aeroguard_single.py")
source = path.read_text(encoding="utf-8")

old_before = '''        # NEW: keep forwarding subsequent observations after a trigger.
        append_post_trigger_telemetry(record)

        ai_history.append(record)
'''

new_before = '''        ai_history.append(record)
'''

old_after = '''        run_ai()
'''

new_after = '''        # Run AI first so a newly triggered observation is included.
        # The finally block also logs it if AI processing raises an error.
        try:
            run_ai()
        finally:
            append_post_trigger_telemetry(record)
'''

if source.count(old_before) != 1:
    raise SystemExit(
        "STOP: Could not uniquely locate the existing telemetry block. "
        "No changes made."
    )

if source.count(old_after) != 1:
    raise SystemExit(
        "STOP: Could not uniquely locate run_ai() in the main loop. "
        "No changes made."
    )

source = source.replace(old_before, new_before, 1)
source = source.replace(old_after, new_after, 1)

path.write_text(source, encoding="utf-8")
print("PATCH APPLIED: triggering observations will be logged.")
