# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information ---------------------------------------------------

_info_ = {}
with open("../../bayspec/__info__.py", "r") as f:
    exec(f.read(), _info_)

project = 'bayspec'
copyright = '2024, Jun Yang'
author = 'Jun Yang'
release = _info_['__version__']

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinxcontrib.rawfiles',
    'myst_nb',
]

autodoc_mock_imports = [
    'xspec_models_cxc'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
html_extra_path = ['_static']
html_css_files = ['custom.css']
