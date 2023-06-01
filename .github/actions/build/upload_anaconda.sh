#!/usr/bin/env bash

set -e
set -x

export PATH=$CONDA/bin:$PATH
conda create -n venv -y python="$(python -c 'import platform; print(platform.python_version())')"
source activate venv
conda install -y anaconda-client
anaconda -t "$ANACONDA_TOKEN" upload --force -u cobyqa dist/artifact/*
