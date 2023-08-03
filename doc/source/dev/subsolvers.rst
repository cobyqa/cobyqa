.. module:: cobyqa.subsolvers

Subproblem solvers
==================

This module implements the subproblem solvers of COBYQA.

.. currentmodule:: cobyqa.subsolvers

The trust-region subproblems, i.e., the normal and tangential Byrd-Omojokun subproblems, are solved approximately using variations of the truncated conjugate gradient method.
The function below implements these methods.

.. autosummary::
    :toctree: generated/

    normal_byrd_omojokun
    tangential_byrd_omojokun
    constrained_tangential_byrd_omojokun

The geometry-improving subproblems are solved approximately using techniques developed by Powell for his solver BOBYQA :cite:`ds-Powell_2009`.
The functions below implement these techniques.

.. autosummary::
    :toctree: generated/

    cauchy_geometry
    spider_geometry

.. bibliography::
    :labelprefix: DS
    :keyprefix: ds-
