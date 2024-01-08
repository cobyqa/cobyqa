.. _installation:

Installation
============

Recommended installation
------------------------

We highly recommend installing COBYQA via `PyPI <https://pypi.org/project/cobyqa>`_.
This does not need you to download the source code.
Install `pip <https://pip.pypa.io/en/stable/installation/>`_ in your system, then execute

.. code-block:: bash

    pip install cobyqa

in a command shell (e.g., the terminal in Linux or Mac, or the Command Shell for Windows).
If your pip launcher is not ``pip``, adapt the command (it may be ``pip3`` for example).
If this command runs successfully, COBYQA is installed.
You may verify whether COBYQA is successfully installed by executing

.. code-block:: bash

    python -c "import cobyqa; cobyqa.show_versions()"

If your Python launcher is not ``python``, adapt the command (it may be ``python3`` for example).

Alternative installation (using source distribution)
----------------------------------------------------

Alternatively, although discouraged, COBYQA can be installed from the source code.
Download and decompress the `source code package <https://github.com/cobyqa/cobyqa/archive/refs/heads/main.zip>`_.
You will obtain a folder containing ``pyproject.toml``.
In a command shell, change your directory to this folder, and then run

.. code-block:: bash

    pip install .
