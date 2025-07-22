import subprocess
from pathlib import Path
import importlib.util
import os
import shutil

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
    "autoapi.extension",
    "myst_parser",
]

autoapi_type = "python"
autoapi_keep_files = True

autoapi_file_patterns = ["*.py", "*.pyi"]
autoapi_dirs = ["./stub/scaluq"]
autoapi_add_toctree_entry = True
autoapi_python_class_content = "class"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autodoc_typehints = "description"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

html_theme = "sphinx_rtd_theme"

# `version` is only used for local build.
# On Read the Docs, the latest version is `latest`` and the specific version
# is the Git tag name.
