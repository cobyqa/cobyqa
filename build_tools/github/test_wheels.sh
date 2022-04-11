#!/usr/bin/env bash

set -e
set -x

# Test the wheel distribution
python3 -m pip install --progress-bar=off pytest
python3 -c "import cobyqa; cobyqa.show_versions()"
python3 -m pytest --pyargs cobyqa
