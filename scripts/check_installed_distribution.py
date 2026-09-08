"""Verify the installed wheel from outside the checkout, including examples."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

root = Path(__file__).resolve().parents[1]
expected_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
environment = dict(os.environ)
environment.pop("PYTHONPATH", None)

with tempfile.TemporaryDirectory(prefix="skgrad-wheel-check-") as directory:
    location = Path(directory)
    shutil.copytree(root / "tests", location / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "examples", location / "examples", ignore=shutil.ignore_patterns("__pycache__"))
    # Release-guard tests exercise repository tooling alongside the wheel.
    (location / "scripts").mkdir()
    shutil.copy2(root / "scripts/check_release_version.py", location / "scripts")
    subprocess.run(
        [sys.executable, "-c",
         "import pathlib, sys, skgrad; "
         "assert skgrad.__version__ == sys.argv[1]; "
         "assert not pathlib.Path(skgrad.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[2]).resolve()); "
         "print('Installed wheel:', skgrad.__version__)",
         expected_version, str(root)],
        cwd=location, env=environment, check=True,
    )
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                   cwd=location, env=environment, check=True)
    for example in sorted((location / "examples").glob("*.py")):
        print(f"Installed-wheel example: {example.name}", flush=True)
        subprocess.run([sys.executable, str(example)],
                       cwd=location, env=environment, check=True)
