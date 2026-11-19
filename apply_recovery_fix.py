from pathlib import Path
import ast

path = Path("aeroguard_single.py")
source = path.read_text(encoding="utf-8")

old = """        process_response(response)

except KeyboardInterrupt:
"""

new = """        try:
            process_response(response)
        except Exception as exc:
            print(f"Telemetry processing failed; monitoring will continue: {exc}")
            continue

except KeyboardInterrupt:
"""

if source.count(old) != 1:
    raise SystemExit("STOP: Expected monitoring-loop section not found exactly once. No changes applied.")

updated = source.replace(old, new, 1)
ast.parse(updated)
path.write_text(updated, encoding="utf-8")
print("PASS: Recovery patch applied and syntax validated.")
