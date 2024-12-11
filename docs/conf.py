# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project's source directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'kinase-library'
copyright = '2024, Tomer Yaron-Barir'
author = 'Tomer Yaron-Barir'
release = 'v1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',         # Automatically document modules/classes/functions
    'sphinx.ext.napoleon',        # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',        # Add links to highlighted source code
    'sphinx.ext.autosummary',     # Generate summary tables for modules/classes
    'sphinx_autodoc_typehints',   # Show type hints in documentation
]

# Enable automatic generation of summary tables
autosummary_generate = True

# Paths for templates and static files
templates_path = ['_templates']
html_static_path = ['_static']

# Patterns to exclude when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    'members': True,             # Document members of classes and modules
    'undoc-members': True,       # Include undocumented members
    'private-members': False,    # Exclude private members (start with '_')
    'show-inheritance': True,    # Show class inheritance
}

# Napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
