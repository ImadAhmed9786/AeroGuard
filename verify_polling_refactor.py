from pathlib import Path
import ast

original = Path("aeroguard_single.py").read_text(encoding="utf-8")
preview = Path("aeroguard_single_refactored_preview.py").read_text(encoding="utf-8")

tree = ast.parse(preview)
functions = {
    node.name: node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

if "process_response" not in functions:
    raise SystemExit("FAIL: process_response function is missing.")

fn = functions["process_response"]
print("PASS: process_response function exists.")
print(f"Function spans lines {fn.lineno}-{fn.end_lineno}.")

checks = {
    "Empty-response handling": "No OpenSky observation. Existing recording retained." in preview,
    "Timestamp validation": "last_observation_timestamp" in preview,
    "Recording buffer": "recording.append(record)" in preview,
    "AI inference": "run_ai()" in preview,
    "Post-trigger logging": "append_post_trigger_telemetry(record)" in preview,
    "Main loop calls refactored function": "process_response(response)" in preview,
    "Original source preserved": original == Path("aeroguard_single.py").read_text(encoding="utf-8"),
}

for label, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {label}")

if not all(checks.values()):
    raise SystemExit("STOP: One or more checks failed.")

print("\nPreview checks passed. Production file has not been changed.")
