import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_installed_version

project = "fbscatnet"
copyright = "2026, Marcel Venturotti"
author = "Marcel Venturotti"
try:
    release = get_installed_version("fbscatnet")
except PackageNotFoundError:
    release = "0.0.0"

sys.path.insert(0, os.path.abspath("../src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Adds links to highlighted source code
    "sphinx.ext.intersphinx",  # Links to external documentation (e.g., Python standard library)
    "sphinx.ext.autosummary",  # Generates stub files for autodoc automatically
]

# --- Autodoc Customisation ---
# Controls how members (functions, classes) are ordered in the docs
autodoc_member_order = "bysource"  # Options: 'alphabetical', 'bysource', 'groupwise'

# Default flags used by all autodoc directives (e.g., show member type hints)
autodoc_typehints = "description"  # Options: 'signature', 'description', 'none', 'both'

# Automatically mock imports that might not be installed in the readthedocs/build environment
autodoc_mock_imports = []

# --- Intersphinx Mapping ---
# Allows linking to Python's built-in types and other libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "sphinx_rtd_theme"

autosummary_generate = True
