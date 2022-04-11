#!/usr/bin/env bash

set -e
set -x

# Create a virtual environment outside of the source folder
cd ../../
python3 -m venv build_env
source build_env/bin/activate

# Build and check the source distribution
cd cobyqa/cobyqa
python3 -m pip install --progress-bar=off numpy scipy cython
python3 -m pip install --progress-bar=off twine
python3 setup.py sdist
python3 -m twine check dist/*.tar.gz
