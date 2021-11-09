#!/usr/bin/env bash

set -e
set -x

# Create a virtual environment outside of the source folder
cd ../../
python -m venv build_env
source build_env/bin/activate

# Build the wheel distribution
cd cobyqa/cobyqa
python -m pip install --progress-bar=off oldest-supported-numpy scipy
python -m pip install --progress-bar=off twine wheel
python setup.py bdist_wheel
python -m twine check dist/*.whl