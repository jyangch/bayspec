# Configuration file for the Sphinx documentation builder.

# -- Project information --

_info_ = {}
with open('../../bayspec/__info__.py') as f:
    exec(f.read(), _info_)

project = 'bayspec'
copyright = '2024, Jun Yang'
author = 'Jun Yang'
release = _info_['__version__']

# -- General configuration --

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinxcontrib.rawfiles',
    'sphinx_copybutton',
    'myst_nb',
]

autodoc_mock_imports = ['xspec_models_cxc']
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

napoleon_use_ivar = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output --

html_theme = 'furo'
html_title = f'BaySpec {release}'
html_static_path = ['_static']
html_extra_path = ['_static']
html_css_files = ['custom.css']

pygments_style = 'friendly'
pygments_dark_style = 'monokai'

html_theme_options = {
    'navigation_with_keys': True,
    'top_of_page_buttons': ['view', 'edit'],
    'light_css_variables': {
        'color-brand-primary': '#1a3a6e',
        'color-brand-content': '#1f5fb3',
        'color-code-background': '#f5f5f5',
    },
    'dark_css_variables': {
        'color-brand-primary': '#7fb3ff',
        'color-brand-content': '#7fb3ff',
        'color-code-background': '#1e1e1e',
    },
}
