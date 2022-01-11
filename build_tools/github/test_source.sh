#!/usr/bin/env bash

set -e
set -x

# Create a virtual environment outside of the source folder
cd ../../
python -m venv test_env
source test_env/bin/activate

# Install and test the source distribution
python -m pip install --progress-bar=off cobyqa/cobyqa/dist/*.tar.gz
python -m pip install --progress-bar=off pytest
python -c "import cobyqa; cobyqa.show_versions()"
python -m pytest --pyargs cobyqa
