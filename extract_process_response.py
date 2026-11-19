from pathlib import Path
import ast

source = Path("aeroguard_single.py").read_text(encoding="utf-8")
tree = ast.parse(source)

function = next(
    (node for node in tree.body
     if isinstance(node, ast.FunctionDef) and node.name == "process_response"),
    None
)

if function is None:
    raise SystemExit("FAIL: process_response not found.")

Path("process_response_test_extract.py").write_text(
    ast.unparse(function) + "\n",
    encoding="utf-8"
)

print("PASS: Extracted process_response() directly from production code.")
print("Next: test empty, stale, duplicate and valid telemetry responses.")
