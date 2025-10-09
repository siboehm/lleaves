#!/usr/bin/env bash
set -euox pipefail

export PYTHON_VERSION=$1

# Install the package in development mode with test and benchmark dependencies
uv pip install --system -e ".[test,benchmark]"

./benchmarks/data/setup_data.sh
pytest -v tests

# Check documentation build only in one job
if [ "${PYTHON_VERSION}" = "3.9" ]; then
  # Install documentation dependencies
  uv pip install --system -e ".[docs]"

  pushd docs
  make html
  popd
fi