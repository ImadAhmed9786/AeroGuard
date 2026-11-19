from pathlib import Path
import ast

path = Path("aeroguard_single.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)

node = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef)
     and n.name == "trigger_test_emergency"),
    None
)
if node is None:
    raise SystemExit("Function not found. No changes made.")

start = node.lineno - 1
end = node.end_lineno
lines = source.splitlines(keepends=True)

replacement = '''def trigger_test_emergency():
    global test_emergency, emergency_triggered

    if not test_emergency or emergency_triggered:
        return

    test_emergency = False
    emergency_triggered = True

    print("\\\\n" + "=" * 60)
    print("TEST EMERGENCY ACTIVATED")
    print("=" * 60)

    try:
        recording_file = save_previous_recording()
        issues = ["TEST EMERGENCY"]

        if recording_file is None:
            issues.append(
                "Recording save failed or no observations were available; "
                "the emergency report will not contain the recording."
            )

        try:
            transmission_ok = activate_transmitter(
                aircraft_id=AIRCRAFT_ID,
                callsign=CALLSIGN,
                issues=issues,
                recording=recording_file,
            )
        except Exception as error:
            transmission_ok = False
            print(f"TRANSMITTER ERROR: {error}")

        print("TEST EMERGENCY COMPLETE.")
        print(
            f"Transmitter result: "
            f"{'SUCCESS' if transmission_ok else 'FAILED'}"
        )
        print("This was a simulation, not a real ATC transmission.")

    except Exception as error:
        print(f"TEST EMERGENCY HANDLER ERROR: {error}")
        print("Test emergency remains latched to prevent duplicate triggers.")
'''

updated = "".join(lines[:start]) + replacement + "".join(lines[end:])
ast.parse(updated)
path.write_text(updated, encoding="utf-8")
print("PATCH APPLIED: manual test emergency has exception handling.")
