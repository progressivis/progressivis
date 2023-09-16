#!/usr/bin/env python3
# type: ignore
# -*- coding: utf-8 -*-
#
# progressivis documentation build configuration file, created by
# sphinx-quickstart on Fri Feb 16 00:36:48 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import sys
import os

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
sys.path.append(os.path.abspath("./_ext"))
sys.path.append(os.path.abspath("./_params"))
extensions = [
    "sphinxcontrib.mermaid",
    "progressivis_mmd",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    # "sphinx_gallery.gen_gallery",
]

myst_heading_anchors = 3  # Generate anchors in headers for hyperlinks

"""
sphinx_gallery_conf = {
    "doc_module": "progressivis",
    "examples_dirs": "./examples",   # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}
"""
# autodoc_typehints = "none"
autodoc_member_order = 'bysource'
autodoc_class_signature = "separated"
napoleon_preprocess_types = True
from progressivis_doc_params import napoleon_type_aliases  # noqa
napoleon_use_param = True
napoleon_custom_sections = [("Module Parameters", 'params_style'), ('Input slots', 'params_style'), ('Output slots', 'params_style')]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = [".rst", ".md"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "progressivis"
copyright = "2018-2023, Inria, Jean-Daniel Fekete and the ProgressiVis contributors"
authors = "Jean-Daniel Fekete, Romain Primet, Christian Poli"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import progressivis  # noqa
version = progressivis.__version__
progressivis.napoleon_type_aliases = napoleon_type_aliases
# The full version, including alpha/beta/rc tags.
release = progressivis.__version__
# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
# language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'templates', 'include', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/progressivis/progressivis",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ]
}

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "progressivis",
    "github_repo": "progressivis",
    "github_version": "master",
    "doc_path": "doc",
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'progressivis'

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'progressivisdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'progressivis.tex', 'progressivis Documentation',
     'Jean-Daniel Fekete', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'progressivis', 'progressivis Documentation',
     [authors], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'progressivis', 'progressivis Documentation',
     authors, 'progressivis', 'One line description of project.',
     'Miscellaneous'),
]


# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    # 'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'pyarrow': ('https://arrow.apache.org/docs', None),
    'datashape': ('https://datashape.readthedocs.io/en/latest/', None),
}

graphviz_output_format = 'svg'
