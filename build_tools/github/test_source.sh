#!/usr/bin/env bash

set -e
set -x

# Create a virtual environment outside of the source folder
cd ../../
python3 -m venv test_env
source test_env/bin/activate

# Install and test the source distribution
python3 -m pip install --progress-bar=off cobyqa/cobyqa/dist/*.tar.gz
python3 -m pip install --progress-bar=off pytest
python3 -c "import cobyqa; cobyqa.show_versions()"
python3 -m pytest --pyargs cobyqa
