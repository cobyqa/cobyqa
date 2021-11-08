#!/usr/bin/env bash

set -e
set -x

# Create a virtual environment outside of the source folder
cd ../../
python -m venv build_env
source build_env/bin/activate

# Build and check the source distribution
cd cobyqa/cobyqa
python -m pip install --progress-bar=off numpy scipy
python -m pip install --progress-bar=off twine
python setup.py sdist
python -m twine check dist/*.tar.gz