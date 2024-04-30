import subprocess

subprocess.run("mkdir -p stub", shell = True, check = True)
subprocess.run("cp ../../python/scaluq/scaluq_core.pyi ./stub/scaluq.py", shell = True, check = True)
subprocess.run("sed -i 's/scaluq.scaluq_core/scaluq/g' ./stub/scaluq.py", shell = True, check=True)

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
autoapi_dirs = ["./stub"]
autoapi_add_toctree_entry = True

autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"

# `version` is only used for local build.
# On Read the Docs, the latest version is `latest`` and the specific version
# is the Git tag name.
