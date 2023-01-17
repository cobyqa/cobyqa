COBYQA: a derivative-free method for general nonlinear optimization
###################################################################

.. image:: https://img.shields.io/github/actions/workflow/status/cobyqa/cobyqa/build.yml?logo=github&style=for-the-badge
    :target: https://github.com/cobyqa/cobyqa/actions/workflows/build.yml

.. image:: https://img.shields.io/readthedocs/cobyqa/latest?logo=readthedocs&style=for-the-badge
    :target: https://www.cobyqa.com/

.. image:: https://img.shields.io/codecov/c/github/cobyqa/cobyqa?logo=codecov&style=for-the-badge
    :target: https://codecov.io/gh/cobyqa/cobyqa/

.. image:: https://img.shields.io/pypi/v/cobyqa?logo=pypi&style=for-the-badge
    :target: https://pypi.org/project/cobyqa/

.. image:: https://img.shields.io/pypi/dm/cobyqa?logo=pypi&style=for-the-badge
    :target: https://pypi.org/project/cobyqa/

.. image:: https://img.shields.io/pypi/pyversions/cobyqa?logo=python&style=for-the-badge
    :target: https://pypi.org/project/cobyqa/

.. image:: https://img.shields.io/pypi/l/cobyqa?style=for-the-badge
    :target: https://github.com/cobyqa/cobyqa/blob/main/LICENSE

COBYQA, named after *Constrained Optimization BY Quadratic Approximations*, is a derivative-free solver for general nonlinear optimization.
It can handle unconstrained, bound-constrained, linearly constrained, and nonlinearly constrained problems.
It uses only function values of the objective function and nonlinear constraint functions, if any.
No derivative information is needed.

The current release of COBYQA is distributed under the BSD-3-Clause license.

**Documentation:** https://www.cobyqa.com/

Installation
============

COBYQA can be installed on `Python 3.8 or above <https://www.python.org>`_.

Dependencies
------------

The following Python packages are required by COBYQA:

* `NumPy <https://www.numpy.org>`_ 1.17.0 or higher, and
* `SciPy <https://www.scipy.org>`_ 1.1.0 or higher.

User installation
-----------------

The easiest way to install COBYQA is using ``pip`` ::

    python -m pip install cobyqa

To check your installation, you can execute ::

    python -c "import cobyqa; cobyqa.show_versions()"

Testing
-------

To execute the test suite of COBYQA, you first need to install the package as ::

    python -m pip install cobyqa[tests]

You can then run the test suite by executing ::

    python -m pytest --pyargs cobyqa

Examples
--------

The folder ``examples`` contains a few examples of how to use COBYQA.
To run ``powell2015.py``, you first need to install the package as ::

    python -m pip install cobyqa[examples]

This will install ``matplotlib`` alongside COBYQA.

Support
=======

To report a bug or request a new feature, please open a new issue using the `issue tracker <https://github.com/cobyqa/cobyqa/issues>`_.
