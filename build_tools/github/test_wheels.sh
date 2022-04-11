#!/usr/bin/env bash

set -e
set -x

# Test the wheel distribution
if [[ $(python -c "import sys; print(sys.version_info[0])") -lt 3 ]]; then
    alias python="python3"
fi
python -m pip install --progress-bar=off pytest
python -c "import cobyqa; cobyqa.show_versions()"
python -m pytest --pyargs cobyqa
