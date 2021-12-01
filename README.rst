#####################################################################
COBYQA: derivative-free solver for nonlinear constrained optimization
#####################################################################

.. image:: https://readthedocs.org/projects/cobyqa/badge/?version=latest
    :target: https://cobyqa.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/ragonneau/cobyqa/actions/workflows/codeql.yml/badge.svg
    :target: https://github.com/ragonneau/cobyqa/actions/workflows/codeql.yml

.. image:: https://github.com/ragonneau/cobyqa/actions/workflows/wheels.yml/badge.svg
    :target: https://github.com/ragonneau/cobyqa/actions/workflows/wheels.yml

.. image:: https://codecov.io/gh/ragonneau/cobyqa/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/ragonneau/cobyqa

.. image:: https://img.shields.io/badge/license-BSD-blue
    :target: https://github.com/ragonneau/cobyqa/blob/main/LICENSE

.. image:: https://shields.io/pypi/v/cobyqa.svg
    :target: https://pypi.org/project/cobyqa/

COBYQA, named after *Constrained Optimization BY Quadratic Approximation*, is a
Python package for solving derivative-free optimization problems under general
nonlinear inequality and equality constraints and is distributed under the
BSD-3-Clause license.

**Documentation:** https://cobyqa.readthedocs.io/

Installation
============

COBYQA can be installed on `Python 3.6 or above <https://www.python.org>`_.

Dependencies
------------

The following Python packages are required by COBYQA:

* `NumPy <https://www.numpy.org>`_ 1.13.3 or higher, and
* `SciPy <https://www.scipy.org>`_ 1.1.0 or higher.

User installation
-----------------

The easiest way to install COBYQA is using ``pip`` ::

    python -m pip install -U cobyqa

To check your installation, you can execute ::

    python -c "import cobyqa; cobyqa.show_versions()"

Testing
-------

After installing `pytest <https://docs.pytest.org>`_ 5.0.1 or higher, you can
execute the test suite of COBYQA via ::

    python -m pytest --pyargs cobyqa

Support
=======

To report a bug or request a new feature, please open a new issue using the
`issue tracker <https://github.com/ragonneau/cobyqa/issues>`_.
