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
     - Initial release.
