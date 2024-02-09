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
:Authors: `Tom M. Ragonneau <https://tomragonneau.com>`_ | `Zaikun Zhang <https://www.zhangzk.net>`_

COBYQA is a derivative-free optimization solver designed to supersede `COBYLA <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html>`_.
Using only functions values, and no derivatives, it aims at solving problems of the form

.. math::

    \min_{x \in \mathcal{X}} \quad f ( x ) \quad \text{s.t.} \quad
    \begin{cases}
        b_l \le A x \le b_u,\\
        c_l \le c ( x ) \le c_u,
    \end{cases}

where :math:`\mathcal{X} = \{ x \in \mathbb{R}^n : l \le x \le u \}`.
COBYQA always respect the bound constraints throughout the optimization process.
Hence, the nonlinear functions :math:`f` and :math:`c` do not need to be well-defined outside :math:`\mathcal{X}`.
In essence, COBYQA is a derivative-free trust-region SQP method based on quadratic models obtained by underdetermined interpolation.
For a more detailed description of the algorithm, see the :ref:`framework description <framework>`.

To install COBYQA using ``pip``, run in your terminal

.. code-block:: bash

    pip install cobyqa

If you are using ``conda``, you can install COBYQA from the `conda-forge <https://anaconda.org/conda-forge/cobyqa>`_ channel by running

.. code-block:: bash

    conda install conda-forge::cobyqa

For more details on the installation and the usage of COBYQA, see the :ref:`user guide <user>`.

Citing COBYQA
-------------

If you would like to acknowledge the significance of COBYQA in your research, we suggest citing the project as follows.

- T.\  M.\  Ragonneau. "Model-Based Derivative-Free Optimization Methods and Software." PhD thesis. Hong Kong, China: Department of Applied Mathematics, The Hong Kong Polytechnic University, 2022. URL: https://theses.lib.polyu.edu.hk/handle/200/12294.
- T.\  M.\  Ragonneau and Z.\  Zhang. COBYQA Version |release|. |year|. URL: https://www.cobyqa.com.

The corresponding BibTeX entries are given hereunder.

.. code-block:: bib
    :substitutions:

    @phdthesis{rago_thesis,
        title        = {Model-Based Derivative-Free Optimization Methods and Software},
        author       = {Ragonneau, T. M.},
        school       = {Department of Applied Mathematics, The Hong Kong Polytechnic University},
        address      = {Hong Kong, China},
        year         = 2022,
        url          = {https://theses.lib.polyu.edu.hk/handle/200/12294},
    }

    @misc{razh_cobyqa,
        author       = {Ragonneau, T. M. and Zhang, Z.},
        title        = {{COBYQA} {V}ersion |release|},
        year         = |year|,
        url          = {https://www.cobyqa.com},
    }

Statistics
----------

As of |today|, COBYQA has been downloaded |total_downloads| times, including

- |pypi_downloads| times on `PyPI <https://pypistats.org/packages/cobyqa>`_ (`mirror downloads <https://pypistats.org/faqs>`_ excluded), and
- |conda_downloads| times on `conda-forge <https://anaconda.org/conda-forge/cobyqa>`_.

The following figure shows the cumulative downloads of COBYQA.

.. plot::

    import json
    from datetime import date, datetime
    from urllib.request import urlopen

    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Download the raw statistics from GitHub.
    base_url = "https://raw.githubusercontent.com/cobyqa/stats/main/archives/"
    conda = json.loads(urlopen(base_url + "conda.json").read())
    pypi = json.loads(urlopen(base_url + "pypi.json").read())

    # Keep only the mirror-excluded statistics for PyPI.
    pypi = [{"date": d["date"], "downloads": d["downloads"]} for d in pypi if d["category"] == "without_mirrors"]

    # Combine the daily statistics into a single list.
    download_dates = []
    daily_downloads = []
    for src in [conda, pypi]:
        for d in src:
            download_date = datetime.strptime(d["date"], "%Y-%m-%d").date()
            try:
                # If the date is already in the list, add the downloads.
                i = download_dates.index(download_date)
                daily_downloads[i] += d["downloads"]
            except ValueError:
                # Otherwise, add the date and downloads.
                download_dates.append(download_date)
                daily_downloads.append(d["downloads"])
    daily_downloads = [d for _, d in sorted(zip(download_dates, daily_downloads))]
    download_dates = sorted(download_dates)
    cumulative_downloads = [sum(daily_downloads[:i]) for i in range(1, len(daily_downloads) + 1)]

    # Plot the cumulative downloads.
    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.plot(download_dates, cumulative_downloads)
    ax.set_xlim(date(2023, 1, 9), download_dates[-1])
    ax.set_ylim(0, cumulative_downloads[-1])
    ax.set_title("Cumulative downloads of COBYQA")
    plt.show()

Acknowledgments
---------------

This work was partially supported by the `Research Grants Council <https://www.ugc.edu.hk/eng/rgc/>`_ of Hong Kong under Grants PF18-24698, PolyU 253012/17P, PolyU 153054/20P, PolyU 153066/21P, and `The Hong Kong Polytechnic University <https://www.polyu.edu.hk/>`_.
