COBYQA documentation
====================

.. toctree::
    :maxdepth: 1
    :hidden:

    User guide <user/index>
    API reference <ref/index>
    Developer guide <dev/index>

:Version: |version|
:Useful links: `Issue tracker <https://github.com/cobyqa/cobyqa/issues>`_ | `Mailing list <https://mail.python.org/mailman3/lists/cobyqa.python.org/>`_
:Authors: `Tom M. Ragonneau <https://tomragonneau.com/>`_ | `Zaikun Zhang <https://www.zhangzk.net/>`_

COBYQA is a derivative-free optimization solver designed to succeed `COBYLA <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html>`_.
Using only functions values, and no derivatives, it aims at solving problems of the form

.. math::

    \min_{x \in \xbd} \quad \obj ( x ) \quad \text{s.t.} \quad
    \begin{cases}
        \aub x \le \bub, ~ \aeq x = \beq,\\
        \cub ( x ) \le 0, ~ \ceq ( x ) = 0,
    \end{cases}

where :math:`\xbd = \{ x \in \R^n : \xl \le x \le \xu \}`.
COBYQA always respect the bound constraints throughout the optimization process.
Hence, the nonlinear functions :math:`\obj`, :math:`\cub`, and :math:`\ceq` do not need to be well-defined outside :math:`\xbd`.
In essence, COBYQA is a derivative-free trust-region SQP method based on quadratic models obtained by underdetermined interpolation.
For a more detailed description of the algorithm, see the :ref:`framework description <framework>`.

To install COBYQA, run in your terminal

.. code-block:: bash

    pip install cobyqa

For more details on the installation and the usage of COBYQA, see the :ref:`user guide <user>`.

Citing COBYQA
-------------

If you would like to acknowledge the significance of COBYQA in your research, we suggest citing the project as follows.

- T.\  M.\  Ragonneau. `Model-Based Derivative-Free Optimization Methods and Software <https://tomragonneau.com/documents/thesis.pdf>`_. PhD thesis, Department of Applied Mathematics, The Hong Kong Polytechnic University, Hong Kong, China, 2022.
- T.\  M.\  Ragonneau and Z.\  Zhang. COBYQA: Constrained Optimization BY Quadratic Approximations. Available at https://www.cobyqa.com, |year|. Version |release|.

The corresponding BibTeX entries are given hereunder.

.. code-block:: bib
    :substitutions:

    @phdthesis{rago_thesis,
        author          = {Ragonneau, T. M.},
        title           = {Model-Based Derivative-Free Optimization Methods and Software},
        school          = {Department of Applied Mathematics, The Hong Kong Polytechnic University},
        address         = {Hong Kong, China},
        year            = 2022,
    }

    @misc{razh_cobyqa,
        author          = {Ragonneau, T. M. and Zhang, Z.},
        title           = {{COBYQA}: {C}onstrained {O}ptimization {BY} {Q}uadratic {A}pproximations},
        howpublished    = {Available at https://www.cobyqa.com},
        note            = {Version |release|},
        year            = |year|,
    }

Statistics
----------

As of |today|, COBYQA has been downloaded |total_downloads| times, including

- |github_downloads| times on `GitHub <https://hanadigital.github.io/grev/?user=cobyqa&repo=cobyqa>`_, and
- |pypi_downloads| times on `PyPI <https://pypistats.org/packages/cobyqa>`_ (`mirror downloads <https://pypistats.org/faqs>`_ excluded).

Acknowledgments
---------------

The early development of COBYQA was funded by the `University Grants Committee <https://www.ugc.edu.hk>`_ of Hong Kong, under the `Hong Kong PhD Fellowship Scheme <https://cerg1.ugc.edu.hk/hkpfs/index.html>`_ (ref. PF18-24698).
It is now supported by `The Hong Kong Polytechnic University <https://www.polyu.edu.hk>`_.
