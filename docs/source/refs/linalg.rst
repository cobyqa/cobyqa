.. module:: cobyqa.linalg

*************************************
Linear algebra (:mod:`cobyqa.linalg`)
*************************************

.. currentmodule:: cobyqa.linalg

This module implements the subproblem solvers of |project|.
They should be investigated for specific purposes only, and users who just wish to solve general derivative-free optimization problems should instead refer to the function `cobyqa.minimize`.

.. autosummary::
    :toctree: generated/

    bvcs
    bvlag
    bvtcg
    cpqp
    lctcg
    nnls