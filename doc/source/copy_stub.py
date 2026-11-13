import ast
import subprocess
from pathlib import Path
import importlib.util
import os

from prepare_stub import prepare_stub

current_dir = Path(__file__).parent.resolve()

stub_dir = Path(current_dir / "stub" / "scaluq/")
stub_dir.mkdir(parents=True, exist_ok=True)

files = []

scaluq_path = os.path.dirname(importlib.util.find_spec("scaluq").origin)
public_stub = Path(scaluq_path) / "__init__.pyi"
public_types = (
    {node.name for node in ast.parse(public_stub.read_text()).body if isinstance(node, ast.ClassDef)}
    if public_stub.is_file()
    else set()
)


def copy_stub_file(target):
    if not os.path.isfile(f"{scaluq_path}/{target}/__init__.pyi"):
        print(f"No stub file found for {target}")
        return
    os.makedirs(f"{stub_dir}/{target}", exist_ok=True)
    Path(f"{stub_dir}/{target}/__init__.pyi").write_text(
        prepare_stub(Path(f"{scaluq_path}/{target}/__init__.pyi").read_text(), public_types),
    )
    files.append(f"{current_dir}/stub/scaluq/{target}/__init__.pyi")


copy_stub_file("")
copy_stub_file("gate")

subprocess.run(["sed", "-i", "/@overload/d"] + files, check=True)
