Release notes
=============

We provide below release notes for the different versions of COBYQA.

.. currentmodule:: cobyqa

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Version
     - Date
     - Remarks
   * - 1.1.2
     - 2024-09-03
     - This is a bugfix release.

       #. Make sure that the disregarded arguments of `scipy.optimize.Bounds`, `scipy.optimize.LinearConstraint`, and `scipy.optimize.NonlinearConstraint` are not passed to the `minimize` function.
   * - 1.1.1
     - 2024-03-12
     - This is a bugfix release.

       #. The objective function values and maximum constraint violations are now those without extreme barriers.
       #. If the returned objective function value or the maximum constraint violation is NaN, the optimization procedure is now considered unsuccessful.
   * - 1.1.0
     - 2024-03-11
     - This is an improvement release.

       #. The computations of the quadratic models have been improved. Instead of using an LBL factorization to solve the KKT conditions, we now employ an eigendecomposition-based method, improving the stability of COBYQA.
       #. Passing unknown constants to the `minimize` function now raises a warning.
   * - 1.0.2
     - 2024-02-08
     - This is a bugfix release.

       #. The returned value of the `minimize` function has been fixed (when a feasible point was encountered, the returned point was not necessarily feasible).
       #. Nonlinear constraints can now be passed as a `dict`.
   * - 1.0.1
     - 2023-01-24
     - This is a bugfix release.

       #. The documentation has been improved.
       #. Typos in the examples have been fixed.
       #. Constants can now be modified by the user.
   * - 1.0.0
     - 2023-01-09
     - This is the initial release.
