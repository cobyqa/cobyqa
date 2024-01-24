.. _dev:

Developer guide
===============

This guide does not cover the usage of COBYQA.
If you want to use COBYQA in your project, please refer to the :ref:`API documentation <api>`.
This guide is intended for developers who want to contribute to the COBYQA solver and to derivative-free optimization solvers in general.

The `cobyqa` module has four submodules, detailed below.
Users do not need to import these submodules when using COBYQA.

.. currentmodule:: cobyqa

The `problem` module implements classes for representing optimization problems.

.. toctree::
    :maxdepth: 2

    problem

The `models` module implements the models used by COBYQA.

.. toctree::
    :maxdepth: 2

    models

The `framework` module implements the trust-region framework used by COBYQA.

.. toctree::
    :maxdepth: 2

    framework

The `subsolvers` module implements the subproblem solvers used by COBYQA.

.. toctree::
    :maxdepth: 2

    subsolvers

The `utils` module implements the utilities for COBYQA.

.. toctree::
    :maxdepth: 2

    utils
