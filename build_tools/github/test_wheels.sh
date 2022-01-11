#!/usr/bin/env bash

set -e
set -x

# Test the wheel distribution
python -m pip install --progress-bar=off pytest
python -c "import cobyqa; cobyqa.show_versions()"
python -m pytest --pyargs cobyqa
