import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(".."))

project = "xDSL"
copyright = "2025"
author = "Mathieu Fehr"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",  # For markdown support
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_rtd_theme"

html_build_dir = os.path.join(os.path.dirname(__file__), "_build/html")
doctrees_dir = os.path.join(os.path.dirname(__file__), "_build/doctrees")

# The suffix of source filenames
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
