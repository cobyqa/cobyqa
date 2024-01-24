#!/usr/bin/env bash

set -e
set -x

# Install cobyqa_latest
find . -type f -maxdepth 1 -exec sed -i "s/cobyqa/cobyqa_latest/g" {} +
find cobyqa -type f -exec sed -i "s/cobyqa/cobyqa_latest/g" {} +
mv cobyqa cobyqa_latest
python -m pip install --progress-bar=off .

# Install cobyqa
python -m pip install --progress-bar=off cobyqa

# Install dependencies
# TODO: Install optiprofiler properly when available on PyPI.
python -m pip install --progress-bar=off numpy
git clone --depth 1 https://github.com/optiprofiler/optiprofiler.git "$GITHUB_WORKSPACE/optiprofiler"
python -m pip install --progress-bar=off "$GITHUB_WORKSPACE/optiprofiler[extra]"

# Download CUTEst and its dependencies
mkdir "$GITHUB_WORKSPACE/cutest"
git clone --depth 1 --branch v2.2.3 https://github.com/ralna/ARCHDefs.git "$GITHUB_WORKSPACE/cutest/archdefs"
git clone --depth 1 --branch v2.1.3 https://github.com/ralna/SIFDecode.git "$GITHUB_WORKSPACE/cutest/sifdecode"
git clone --depth 1 --branch v2.0.42 https://github.com/ralna/CUTEst.git "$GITHUB_WORKSPACE/cutest/cutest"
git clone --depth 1 --branch v0.5 https://bitbucket.org/optrove/sif.git "$GITHUB_WORKSPACE/cutest/mastsif"

# Set the environment variables
export ARCHDEFS="$GITHUB_WORKSPACE/cutest/archdefs"
export SIFDECODE="$GITHUB_WORKSPACE/cutest/sifdecode"
export CUTEST="$GITHUB_WORKSPACE/cutest/cutest"
export MASTSIF="$GITHUB_WORKSPACE/cutest/mastsif"
export MYARCH=pc64.lnx.gfo
{
  echo "ARCHDEFS=$ARCHDEFS"
  echo "SIFDECODE=$SIFDECODE"
  echo "CUTEST=$CUTEST"
  echo "MASTSIF=$MASTSIF"
  echo "MYARCH=$MYARCH"
} >> "$GITHUB_ENV"

# Build and install CUTEst
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh)"

# Add the benchmarks to the PYTHONPATH
echo "PYTHONPATH=$PWD/benchmarks:$PYTHONPATH" >> "$GITHUB_ENV"
