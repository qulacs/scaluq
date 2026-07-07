import subprocess
from pathlib import Path
import importlib.util
import os
import shutil

current_dir = Path(__file__).parent.resolve()

stub_dir = Path(current_dir / "stub" / "scaluq/")
stub_dir.mkdir(parents=True, exist_ok=True)

files = []

scaluq_path = os.path.dirname(importlib.util.find_spec("scaluq").origin)


def copy_stub_file(target):
    if not os.path.isfile(f"{scaluq_path}/{target}/__init__.pyi"):
        print(f"No stub file found for {target}")
        return
    os.makedirs(f"{stub_dir}/{target}", exist_ok=True)
    shutil.copyfile(
        f"{scaluq_path}/{target}/__init__.pyi",
        f"{current_dir}/stub/scaluq/{target}/__init__.pyi",
    )
    files.append(f"{current_dir}/stub/scaluq/{target}/__init__.pyi")


copy_stub_file("")
copy_stub_file("gate")

subprocess.run(["sed", "-i", "/@overload/d"] + files, check=True)
