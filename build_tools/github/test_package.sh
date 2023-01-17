#!/usr/bin/env bash

set -e
set -x

for f in dist/*.tar.gz dist/*.whl; do
    python -m venv venv
    source venv/bin/activate
    python -m pip install --progress-bar=off "$f[tests]"
    python -c "import cobyqa; cobyqa.show_versions()"
    python -m pytest --pyargs cobyqa
    deactivate
    rm -r venv
done
