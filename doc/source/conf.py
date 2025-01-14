import sys
import subprocess
from pathlib import Path

stub_dir = Path('./stub/scaluq/')
stub_dir.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
    '-m', 'scaluq.scaluq_core.f64',
    '-o', './stub/scaluq/f64/__init__.py'])

subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
    '-m', 'scaluq.scaluq_core.f64.gate',
    '-o', './stub/scaluq/f64/gate.py'])

subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
    '-m', 'scaluq.scaluq_core.f32',
    '-o', './stub/scaluq/f32/__init__.py'])

subprocess.run([sys.executable, '-m', 'nanobind.stubgen',
    '-m', 'scaluq.scaluq_core.f32.gate',
    '-o', './stub/scaluq/f32/gate.py'])

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

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'packages': ['base', 'ams', 'braket', 'noerrors', 'noundefined'],
        'processEscapes': True, 
        'processEnvironments': True
    }
}

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

autodoc_typehints = 'none'

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'math_number_all': True,
}
# html_js_files = [
#     'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js'
# ]

# `version` is only used for local build.
# On Read the Docs, the latest version is `latest`` and the specific version
# is the Git tag name.
