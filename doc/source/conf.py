# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import cobyqa


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'COBYQA'
author = 'Tom M. Ragonneau and Zaikun Zhang'
copyright = f'2021\u2013{datetime.now().year}, {author}'

# Short version (including .devX, rcX, b1 suffixes if present).
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', cobyqa.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)

# Full version, including alpha/beta/rc tags.
release = cobyqa.__version__

# Download statistics.
archive = urlopen('https://raw.githubusercontent.com/cobyqa/stats/main/archives/total.json')
downloads = json.loads(archive.read())


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx_design',
    'sphinx_substitution_extensions',
]

templates_path = ['_templates']

today_fmt = '%B %d, %Y'

# The ReST default role to use for all documents.
default_role = 'autolink'

# Whether parentheses are appended to function and method role text.
add_function_parentheses = False

# String to include at the beginning of every source file.
rst_prolog = f'''
.. |downloads_total| replace:: {sum(downloads.values())}
.. |year| replace:: {datetime.now().year}
'''

# Suppress 'WARNING: unknown mimetype for ..., ignoring'.
suppress_warnings = ['epub.unknown_project_files']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_css_files = ['cobyqa.css']

html_theme_options = {
    'logo': {
        'text': project,
    },
    'icon_links': [
        {
            'name': f'GitHub ({downloads["github"]} downloads)',
            'url': 'https://github.com/cobyqa/cobyqa',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': f'PyPI ({downloads["pypi"]} downloads)',
            'url': 'https://pypi.org/project/cobyqa',
            'icon': 'fa-solid fa-box',
        },
    ],
    'switcher': {
        'json_url': 'https://www.cobyqa.com/en/latest/_static/switcher.json',
        'version_match': version,
    },
    'navbar_end': ['version-switcher', 'theme-switcher', 'navbar-icon-links'],
    'navbar_persistent': ['search-button'],
    'navbar_align': 'left',
    # 'announcement': '<p></p>',
}

html_context = {
    'github_user': 'cobyqa',
    'github_repo': 'cobyqa',
    'github_version': 'main',
    'doc_path': 'doc/source',
}

html_title = f'{project} v{version} Manual'

html_static_path = ['_static']

html_copy_source = False


# -- Options for HTML help output ---------------------------------------------

htmlhelp_basename = 'cobyqa'


# -- Math support for HTML outputs --------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            # Extra mathematical functions.
            'card': r'\operatorname{card}',
            'abs': [r'#1\lvert#2#1\rvert', 2, ''],
            'norm': [r'#1\lVert#2#1\rVert', 2, ''],
            'set': [r'#1\{#2#1\}', 2, ''],

            # Sets in blackboard-bold style font.
            'R': r'{\mathbb{R}}',

            # Mathematical operators in sans serif style font.
            'T': r'{\mathsf{T}}',
        }
    }
}


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files.
latex_documents = [
    # Read the Docs do not handle multiple PDF files yet.
    # See https://github.com/readthedocs/readthedocs.org/issues/2045
    # ('algorithms/index', 'cobyqa-user.tex', 'COBYQA Reference', author, 'manual'),
    ('reference/index', 'cobyqa-ref.tex', 'COBYQA User Guide', author, 'manual'),
]

latex_elements = {
    'papersize': 'a4paper',
    'fontenc': r'\usepackage[LGR,T1]{fontenc}',
    'preamble': r'''
\usepackage{dsfont}

% Extra mathematical functions.
\DeclareMathOperator{\card}{card}
\newcommand{\abs}[2][]{#1\lvert#2#1\rvert}
\newcommand{\norm}[2][]{#1\lVert#2#1\rVert}
\newcommand{\set}[2][]{#1\{#2#1\}}

% Sets in blackboard-bold style font.
\newcommand{\R}{\mathds{R}}

% Mathematical operators in sans serif style font.
\newcommand{\T}{\mathsf{T}}

% Increase the default table of content depth.
\setcounter{tocdepth}{2}
    ''',
}


# -- Numpy’s Sphinx extension -------------------------------------------------

numpydoc_use_plots = True


# -- Generate autodoc summaries -----------------------------------------------

autosummary_generate = True


# -- Link to other projects’ documentation ------------------------------------

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://scipy.github.io/devdocs/', None),
}


# -- Add external links to source code ----------------------------------------

def linkcode_resolve(domain, info):
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

    repository = f'https://github.com/cobyqa/cobyqa'
    if 'dev' in release:
        return f'{repository}/blob/main/cobyqa/{fn}{lines}'
    else:
        return f'{repository}/blob/v{release}/cobyqa/{fn}{lines}'
