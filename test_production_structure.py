import ast
from pathlib import Path

source_path = Path("aeroguard_single.py")
source = source_path.read_text(encoding="utf-8-sig")
tree = ast.parse(source, filename=str(source_path))

functions = [
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
]

required = [
    "find_aircraft",
    "save_previous_recording",
    "save_rolling_snapshot",
    "append_post_trigger_telemetry",
    "trigger_emergency",
    "create_features",
    "run_ai",
]

missing = [name for name in required if name not in functions]

print("Production source parsed successfully.")
print("Functions found:", len(functions))

if missing:
    print("Missing functions:", ", ".join(missing))
    raise SystemExit(1)

print("All required production functions found.")
print("Live OpenSky setup was NOT executed.")
print("Production files were NOT modified.")
