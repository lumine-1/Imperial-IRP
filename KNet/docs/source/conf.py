# -- Project information -----------------------------------------------------
project = "KNet"
author = "Yiding Hao"
release = "1.1"

# -- Path setup: make project root importable --------------------------------
# conf.py is in docs/source/, so project root is two levels up.
import os
import sys
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))
# Now `import models`, `import process`, `import utils` should work.

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

autosummary_generate = True

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = False

# Type hints rendering
typehints_fully_qualified = False
autodoc_typehints = "description"

# Default autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

# Mock heavy deps to avoid import errors during doc build
autodoc_mock_imports = ["torch", "torchvision", "numpy", "h5py", "nibabel", "tqdm"]

# Patterns to ignore
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "../../runs/*",
    "../../prepared/*",
    "../../tests/*",
]

# Intersphinx (optional)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "torch": ("https://pytorch.org/docs/stable", {}),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "KNet Documentation"

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "linkify"]
