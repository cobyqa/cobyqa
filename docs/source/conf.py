# Configuration file for the Sphinx documentation builder.
import re
import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.
BASE_DIR = Path('.').resolve(strict=True).parent.parent
sys.path.insert(0, str(BASE_DIR))


# -- Project information ------------------------------------------------------

import cobyqa  # noqa

project = 'COBYQA'
copyright = '2021, Tom M. Ragonneau'
author = 'Tom M. Ragonneau'

# The short version (including .devX, rcX, b1 suffixes if present).
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', cobyqa.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)

# The full version, including alpha/beta/rc tags.
release = cobyqa.__version__


# -- General configuration ----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Format the current date.
today_fmt = '%B %d, %Y'

# The ReST default role to use for all documents.
default_role = 'autolink'

# Whether parentheses are appended to function and method role text.
add_function_parentheses = False


# -- Options for HTML output --------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'collapse_navigation': True,
    'footer_items': ['copyright', 'last-updated', 'sphinx-version'],
    'github_url': 'https://github.com/ragonneau/cobyqa',
    'logo_link': 'index',
}

html_title = f'{project} v{version} Manual'

html_logo = '_static/logo.svg'

html_favicon = '_static/favicon/favicon.ico'

html_static_path = ['_static']

html_last_updated_fmt = '%B %d, %Y'

html_additional_pages = {
    'index': 'index.html',
}

html_copy_source = False


# -- Options for HTML help output ---------------------------------------------

htmlhelp_basename = 'cobyqa'


# -- Options for LaTeX output -------------------------------------------------

# Paper size ('letter' or 'a4').
latex_paper_size = 'a4'

# Font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Use XeLaTeX engine for better support of unicode characters
latex_engine = 'xelatex'

# Grouping the document tree into LaTeX files.
_authors = 'Tom M. Ragonneau'
latex_documents = [
    ('algo/index', 'cobyqa-user.tex', 'COBYQA Reference', _authors, 'manual'),
    ('refs/index', 'cobyqa-ref.tex', 'COBYQA User Guide', _authors, 'manual'),
]

latex_elements = {
    'fontenc': r'\usepackage[LGR,T1]{fontenc}',
    'preamble': r'''
\usepackage{dsfont}

% Mathematical constants, sets, and notations
\def\eu{\ensuremath{\mathrm{e}}}
\def\iu{\ensuremath{\mathrm{i}}}
\def\du{\ensuremath{\mathrm{d}}}
\def\C{\ensuremath{\mathds{C}}}
\def\N{\ensuremath{\mathds{N}}}
\def\Q{\ensuremath{\mathds{Q}}}
\def\R{\ensuremath{\mathds{R}}}
\def\Z{\ensuremath{\mathds{Z}}}
\def\sft{\ensuremath{\mathsf{T}}}

% Dedicated mathematical macros
\newcommand{\dv}{x}
\newcommand{\icon}{i}
\newcommand{\iiter}{k}
\newcommand{\lmv}{\lambda}
\newcommand{\nv}{n}
\newcommand{\tsv}{d}

\newcommand{\bl}{l}
\newcommand{\bu}{u}
\newcommand{\con}[1][\icon]{c_{#1}}
\newcommand{\lag}{\mathcal{L}}
\newcommand{\meq}{n_{\scriptscriptstyle\seq}}
\newcommand{\mub}{n_{\scriptscriptstyle\sub}}
\newcommand{\obj}{f}
\newcommand{\seq}{\mathcal{E}}
\newcommand{\sub}{\mathcal{I}}
    ''',
}

# Image to place at the top of the title page.
latex_logo = '_static/logo.pdf'


# -- Numpy’s Sphinx extensions ------------------------------------------------

numpydoc_use_plots = True


# -- Generate autodoc summaries -----------------------------------------------

autosummary_generate = True


# -- Link to other projects’ documentation ------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}


# -- Math support for HTML outputs --------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            # Mathematical constants, sets, and notations
            'eu': r'{\mathrm{e}}',
            'iu': r'{\mathrm{i}}',
            'du': r'{\mathrm{d}}',
            'C': r'{\mathbb{C}}',
            'N': r'{\mathbb{N}}',
            'Q': r'{\mathbb{Q}}',
            'R': r'{\mathbb{R}}',
            'Z': r'{\mathbb{Z}}',
            'sft': r'{\mathsf{T}}',

            # Dedicated mathematical macros
            'dv': '{x}',
            'icon': '{i}',
            'iiter': '{k}',
            'lmv': r'{\lambda}',
            'nv': '{n}',
            'stv': '{d}',

            'bl': '{l}',
            'bu': '{u}',
            'con': [r'{c_{#1}}', 1, '\icon'],
            'lag': r'{\mathcal{L}}',
            'meq': '{n_{\scriptscriptstyle\seq}}',
            'mub': '{n_{\scriptscriptstyle\sub}}',
            'obj': '{f}',
            'seq': r'{\mathcal{E}}',
            'sub': r'{\mathcal{I}}',
        }
    }
}
