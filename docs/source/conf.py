# Configuration file for the Sphinx documentation builder.
import inspect
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
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Format the current date.
today_fmt = '%B %d, %Y'

# The ReST default role to use for all documents.
default_role = 'autolink'

# Whether parentheses are appended to function and method role text.
add_function_parentheses = False

# Suppress "WARNING: unknown mimetype for ..., ignoring"
suppress_warnings = ['epub.unknown_project_files']


# -- Options for HTML output --------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'collapse_navigation': True,
    'footer_items': ['copyright', 'last-updated', 'sphinx-version'],
    # 'github_url': 'https://github.com/ragonneau/cobyqa',
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/ragonneau/cobyqa',
            'icon': 'fab fa-github-square',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/cobyqa',
            'icon': 'fab fa-python',
        },
    ],
    'logo_link': 'index',
}

html_title = f'{project} v{version} Manual'

html_logo = '_static/cobyqa.svg'

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

% Constants and mathematical functions in Roman style font.
\def\eu{\ensuremath{\mathrm{e}}}
\def\iu{\ensuremath{\mathrm{i}}}
\def\du{\ensuremath{\mathrm{d}}}
\DeclareMathOperator{\vspan}{span}

% Extra mathematical functions
\newcommand{\abs}[2][]{#1\lvert#2#1\rvert}
\newcommand{\ceil}[2][]{#1\lceil#2#1\rceil}
\newcommand{\floor}[2][]{#1\lfloor#2#1\rfloor}
\newcommand{\norm}[2][]{#1\lVert#2#1\rVert}
\newcommand{\set}[2][]{#1\{#2#1\}}
\newcommand{\inner}[2][]{#1\langle#2#1\rangle}

% Sets in blackboard-bold style font.
\def\C{\ensuremath{\mathds{C}}}
\def\N{\ensuremath{\mathds{N}}}
\def\Q{\ensuremath{\mathds{Q}}}
\def\R{\ensuremath{\mathds{R}}}
\def\Z{\ensuremath{\mathds{Z}}}

% Mathematical operators in sans serif style font
\def\T{\ensuremath{\mathsf{T}}}
    ''',
}


# -- Numpy’s Sphinx extensions ------------------------------------------------

numpydoc_use_plots = True


# -- Generate autodoc summaries -----------------------------------------------

autosummary_generate = True


# -- Link to other projects’ documentation ------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}


# -- Add external links to source code ----------------------------------------

def linkcode_resolve(domain, info):
    """
    Get the URL to source code corresponding to the given object.
    """
    if domain != 'py':
        return None

    # Get the object indicated by the module name.
    obj = sys.modules.get(info['module'])
    if obj is None:
        return None
    for part in info['fullname'].split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Strip the decorators of the object.
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    # Get the relative path to the source of the object.
    try:
        fn = Path(inspect.getsourcefile(obj)).resolve(strict=True)
    except TypeError:
        return None
    else:
        fn = fn.relative_to(Path(cobyqa.__file__).resolve(strict=True).parent)

    # Ignore re-exports as their source files are not within the repository.
    module = inspect.getmodule(obj)
    if module is not None and not module.__name__.startswith('cobyqa'):
        return None

    # Get the line span of the object in the source file.
    try:
        source, lineno = inspect.getsourcelines(obj)
        lines = f'#L{lineno}-L{lineno + len(source) - 1}'
    except OSError:
        lines = ''

    repository = 'https://github.com/ragonneau/cobyqa'
    if 'dev' in release:
        return f'{repository}/blob/main/cobyqa/{fn}{lines}'
    else:
        return f'{repository}/blob/v{release}/cobyqa/{fn}{lines}'


# -- Math support for HTML outputs --------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            # Constants and mathematical functions in Roman style font.
            'eu': r'{\mathrm{e}}',
            'iu': r'{\mathrm{i}}',
            'du': r'{\mathrm{d}}',
            'vspan': r'\operatorname{span}',

            # Extra mathematical functions
            'abs': [r'#1\lvert#2#1\rvert', 2, ''],
            'ceil': [r'#1\lceil#2#1\rceil', 2, ''],
            'floor': [r'#1\lfloor#2#1\rfloor', 2, ''],
            'norm': [r'#1\lVert#2#1\rVert', 2, ''],
            'set': [r'#1\{#2#1\}', 2, ''],
            'inner': [r'#1\langle#2#1\rangle', 2, ''],

            # Sets in blackboard-bold style font.
            'C': r'{\mathbb{C}}',
            'N': r'{\mathbb{N}}',
            'Q': r'{\mathbb{Q}}',
            'R': r'{\mathbb{R}}',
            'Z': r'{\mathbb{Z}}',

            # Mathematical operators in sans serif style font
            'T': r'{\mathsf{T}}',
        }
    }
}

# -- Bibliography files and encoding ------------------------------------------

bibtex_bibfiles = [
    '_static/cobyqa-strings.bib',
    '_static/cobyqa-refs.bib',
]

bibtex_encoding = 'latin'

bibtex_default_style = 'plain'

bibtex_cite_id = 'cite-{bibliography_count}-{key}'

bibtex_footcite_id = 'footcite-{key}'

bibtex_bibliography_id = 'bibliography-{bibliography_count}'

bibtex_footbibliography_id = 'footbibliography-{footbibliography_count}'

bibtex_bibliography_header = '''
.. only:: html or text

    .. rubric:: References
'''

bibtex_footbibliography_header = bibtex_bibliography_header
