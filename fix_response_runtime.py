from pathlib import Path
import textwrap

path = Path("test_response_runtime.py")
source = path.read_text(encoding="utf-8-sig")

old = '''    code = "while True:\\n" + "".join(
        "    " + line + "\\n" for line in SOURCE.splitlines()
    ) + "    break\\n"

    exec(compile(code, "<production-response-block>", "exec"), env)
'''

new = '''    block = textwrap.dedent(SOURCE)
    code = "while True:\\n" + textwrap.indent(block, "    ") + "\\n"
    exec(compile(code, "<production-response-block>", "exec"), env)
'''

if old not in source:
    print("Old wrapper not found. Replacing the harness with a clean wrapper.")
    start = source.index("    # The production section contains")
    end = source.index("    return env, valid_state", start)
    source = source[:start] + new + source[end:]
else:
    source = source.replace(old, new)

if "import textwrap" not in source:
    source = source.replace("import time\\n", "import time\\nimport textwrap\\n", 1)

path.write_text(source, encoding="utf-8")
print("PASS: Runtime harness wrapper updated.")
