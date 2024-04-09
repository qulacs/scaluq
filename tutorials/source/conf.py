import os

project = 'scaluq'
copyright = '2024, Fuji Lab.'
author = 'Fuji Lab.'
release = '0.0.1'

extensions = [
    "sphinx.ext.napoleon",
    'autoapi.extension',
    'sphinx.ext.githubpages',
]

autoapi_type = "python"
autoapi_keep_files = True

# The order of `autoapi_file_patterns`` specifies the order of preference for reading files.
# So, we give priority to `*.pyi`.
# https://github.com/readthedocs/sphinx-autoapi/issues/243#issuecomment-684190179
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_dirs = ["../../typings/qulacs2023"]
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

# `version` is only used for local build.
# On Read the Docs, the latest version is `latest`` and the specific version
# is the Git tag name.
