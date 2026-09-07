"""Run every standalone example using the active installed environment."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for example in sorted((root / "examples").glob("*.py")):
    print(f"Running {example.name}", flush=True)
    subprocess.run([sys.executable, str(example)], cwd=root, check=True)
