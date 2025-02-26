import sys

from pathlib import Path
ROOT = Path(__file__).parents[3]
PIPELINE_ROOT = Path.joinpath(ROOT, "src")
sys.path.append(PIPELINE_ROOT.as_posix())
#print(PIPELINE_ROOT)
#sys.exit()

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Brainsharer Pipeline'
copyright = "2023, Kleinfeld lab UCSD"
author = "Kleinfeld lab UCSD"
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',  # Create neat summary tables
]
autosummary_generate = False  # Turn on sphinx.ext.autosummary
autodoc_mock_imports = ["settings", "SimpleITK-SimpleElastix", "cloudvolume", \
    "neuroglancer", "taskqueue", "torch", "igneous", "torchvision"]



templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'emacs'


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/250.png"
html_theme_options = {
    'logo_only': True,
    # Toc options
    'collapse_navigation': True,
    'titles_only': True
}
