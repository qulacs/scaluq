import subprocess

subprocess.run("mkdir stub", shell = True, check = True)
subprocess.run("cp ../../python/qulacs2023/qulacs_core.pyi ./stub/qulacs2023.py", shell = True, check = True)
subprocess.run("sed -i 's/qulacs2023.qulacs_core/qulacs2023/g' ./stub/qulacs2023.py", shell = True, check=True)

project = 'scaluq'
copyright = '2024, Fuji Lab.'
author = 'Fuji Lab.'
release = '0.0.1'

extensions = [
    "sphinx.ext.napoleon",
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
