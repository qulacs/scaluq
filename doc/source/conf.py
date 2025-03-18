import sys
import subprocess
from pathlib import Path

stub_dir = Path('./stub/scaluq/')
stub_dir.mkdir(parents=True, exist_ok=True)

files = []

subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
    '-m', 'scaluq.scaluq_core',
    '-o', './stub/scaluq/__init__.py'])
files.append('./stub/scaluq/__init__.py')
for space in ['default', 'host']:
    subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
        '-m', f'scaluq.scaluq_core.{space}',
        '-o', f'./stub/scaluq/{space}/__init__.py'])
    files.append(f'./stub/scaluq/{space}/__init__.py')
    for precision in ['f16', 'f32', 'f64', 'bf16']:
        subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
            '-m', f'scaluq.scaluq_core.{space}.{precision}',
            '-o', f'./stub/scaluq/{space}/{precision}/__init__.py'])
        files.append(f'./stub/scaluq/{space}/{precision}/__init__.py')
        subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
            '-m', f'scaluq.scaluq_core.{space}.{precision}.gate',
            '-o', f'./stub/scaluq/{space}/{precision}/gate.py'])
        files.append(f'./stub/scaluq/{space}/{precision}/gate.py')

subprocess.run(["sed", "-i", "/@overload/d"] + files, check=True)

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_math_dollar',
    'autoapi.extension',
]

autoapi_type = "python"
autoapi_keep_files = True

autoapi_file_patterns = ["*.py"]
autoapi_dirs = ["./stub/scaluq"]
autoapi_add_toctree_entry = True
autoapi_python_class_content = 'class'

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autodoc_typehints = 'description'

html_theme = "sphinx_rtd_theme"

# `version` is only used for local build.
# On Read the Docs, the latest version is `latest`` and the specific version
# is the Git tag name.
