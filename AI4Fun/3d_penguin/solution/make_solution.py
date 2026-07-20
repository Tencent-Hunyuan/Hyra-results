import json, os

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "build_script.py"), "r", encoding="utf-8") as f:
    src = f.read()

with open(os.path.join(here, "solution.json"), "w", encoding="utf-8") as f:
    json.dump({"build_script": src}, f)

print("wrote solution.json (%d chars of build_script)" % len(src))
