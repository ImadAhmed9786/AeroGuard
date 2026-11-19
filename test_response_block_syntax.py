from pathlib import Path
import ast

source = Path("response_processing_section.txt").read_text(encoding="utf-8")
lines = source.splitlines()

# Remove the original 8-space indentation, then place the block
# inside a function and loop, preserving its internal indentation.
if not lines or not lines[0].startswith("        "):
    raise SystemExit("Unexpected extraction indentation.")

dedented = [
    line[8:] if line.startswith("        ") else line
    for line in lines
]

wrapped = ["def _test_block():"]
wrapped.append("    while True:")
wrapped.extend(
    ("        " + line) if line else ""
    for line in dedented
)
wrapped.append("        break")

code = "\n".join(wrapped) + "\n"
ast.parse(code)
compile(code, "response_processing_section.txt", "exec")

print("PASS: Extracted production block compiles inside a loop.")
print("Continue statements:", sum(line.strip() == "continue" for line in dedented))
print("No production code executed or modified.")
