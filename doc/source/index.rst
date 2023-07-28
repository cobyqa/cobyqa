.. _cobyqa_docs_mainpage:

COBYQA documentation
====================

.. toctree::
    :maxdepth: 1
    :hidden:

    Algorithms <algorithms/index>
    API reference <reference/index>

:Version: |version|
:Useful links: `Issue tracker <https://github.com/cobyqa/cobyqa/issues>`_ | `Mailing list <https://mail.python.org/mailman3/lists/cobyqa.python.org/>`_
:Authors: `Tom M. Ragonneau <https://tomragonneau.com/>`_ | `Zaikun Zhang <https://www.zhangzk.net/>`_

.. hint::

    To install COBYQA for Python, simply run :code:`pip install cobyqa` in your terminal.

COBYQA is designed to succeed `COBYLA <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html>`_ as a derivative-free solver for general nonlinear optimization.
It aims at solving problems of the form

.. math::

    \min_{x \in \R^n} f(x) \quad \text{s.t.} \quad \left\{ \begin{array}{l}
        g(x) \le 0,\\
        h(x) = 0,\\
        l \le x \le u,
    \end{array} \right.

where :math:`f` is a real-valued objective function, :math:`g` and :math:`h` are vector-valued constraint functions, and :math:`l` and :math:`u` are vectors of lower and upper bounds on the variables :math:`x`.
COBYQA uses function values of :math:`f`, :math:`g`, and :math:`h`, but no derivatives.
It also assumes that the bound constraints are inviolable and always respect them.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :img-top: ../source/_static/index/book.svg

        Algorithms
        ^^^^^^^^^^

        This section provides detailed mathematical descriptions of the algorithms underneath COBYQA.

        .. button-ref:: algorithms
            :expand:
            :color: secondary
            :click-parent:

            To the algorithm descriptions

    .. grid-item-card::
        :img-top: ../source/_static/index/gears.svg

        API reference
        ^^^^^^^^^^^^^

        This section references an exhaustive manual detailing the functions, modules, and objects included in COBYQA.

        .. button-ref:: reference
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

COBYQA aims at being a modern successor of the well-known solver COBYLA.

.. note::

    As of |today|, COBYQA has been downloaded |downloads_total| times (mirror downloads excluded).

Citing COBYQA
-------------

If you would like to acknowledge the significance of COBYQA in your research, we suggest citing the project as follows:

- T.\  M.\  Ragonneau. "`Model-Based Derivative-Free Optimization Methods and Software <https://tomragonneau.com/documents/thesis.pdf>`_." Ph.D.\  thesis. Hong Kong: Department of Applied Mathematics, The Hong Kong Polytechnic University, 2022.
- T.\  M.\  Ragonneau and Z.\  Zhang. COBYQA: Constrained Optimization BY Quadratic Approximations. Version |release|. |year|. URL: https://www.cobyqa.com.

The corresponding BibLaTeX entries are given hereunder.

.. code-block:: bib
    :substitutions:

    @thesis{rago22,
        author      = {Ragonneau, T. M.},
        title       = {Model-Based Derivative-Free Optimization Methods and Software},
        type        = {phdthesis},
        institution = {Department of Applied Mathematics, The Hong Kong Polytechnic University},
        location    = {Hong Hong},
        date        = {2022},
    }

    @software{razh22,
        author       = {Ragonneau, T. M. and Zhang, Z.},
        title        = {{COBYQA}: Constrained Optimization BY Quadratic Approximations},
        url          = {https://www.cobyqa.com},
        version      = {|release|},
        date         = {|year|},
    }


Acknowledgments
---------------

The early development of COBYQA was funded by the `University Grants Committee <https://www.ugc.edu.hk/>`_ of Hong Kong, under the `Hong Kong Ph.D. Fellowship Scheme <https://cerg1.ugc.edu.hk/hkpfs/index.html>`_ (ref. PF18-24698).
It is now supported by `The Hong Kong Polytechnic University <https://www.polyu.edu.hk/>`_.
