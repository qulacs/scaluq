import sys
import subprocess

subprocess.run([sys.executable, '-m', 'nanobind.stubgen', '-m', 'scaluq.scaluq_core', '-o', './stub/scaluq/__init__.py'])
subprocess.run([sys.executable, '-m', 'nanobind.stubgen', '-m', 'scaluq.scaluq_core.gate', '-o', './stub/scaluq/gate.py'])

project = 'scaluq'
copyright = '2024, Fuji Lab.'
author = 'Fuji Lab.'
release = '0.0.1'

extensions = [
    "sphinx.ext.napoleon",
    'sphinx.ext.mathjax',
    'sphinx_math_dollar',
    'autoapi.extension',
]

autoapi_type = "python"
autoapi_keep_files = True

autoapi_file_patterns = ["*.py"]
autoapi_dirs = ["./stub/scaluq"]
autoapi_add_toctree_entry = True

autoapi_python_class_content = 'both'
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

html_theme = "sphinx_rtd_theme"

# `version` is only used for local build.
# On Read the Docs, the latest version is `latest`` and the specific version
# is the Git tag name.
