import ast
from pathlib import Path

tree = ast.parse(
    "def wrapper():\n    while True:\n" +
    "".join(
        "        " + line if line.strip() else "\n"
        for line in Path("response_processing_section.txt")
        .read_text(encoding="utf-8").splitlines(keepends=True)
    )
)

names = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        names.add(node.id)

print("Names read by extracted production block:")
for name in sorted(names):
    print(" ", name)
