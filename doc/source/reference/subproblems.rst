.. module:: cobyqa.subproblems

Subproblem solvers (:mod:`cobyqa.subproblems`)
==============================================

.. currentmodule:: cobyqa.subproblems

This module implements the subproblem solvers of COBYQA.
They should be investigated for specific purposes only, and users who just wish to solve general derivative-free optimization problems should instead refer to the function `cobyqa.minimize`.

.. autosummary::
    :toctree: generated/

    bound_constrained_cauchy_step
    bound_constrained_normal_step
    bound_constrained_tangential_step
    bound_constrained_xpt_step
    linearly_constrained_tangential_step
