#!/usr/bin/env bash
set -euox pipefail

export PYTHON_VERSION=$1

# Install the package in development mode with test and benchmark dependencies
uv pip install --system -e ".[test,benchmark]"

./benchmarks/data/setup_data.sh
pytest -v tests

# Check documentation build only in one job, also do releases
if [ "${PYTHON_VERSION}" = "3.9" ]; then
  # Install documentation dependencies
  uv pip install --system -e ".[docs]"

  pushd docs
  make html
  popd

  # Install build dependencies and build the package
  uv pip install --system build
  python -m build --sdist
  python -m build --wheel
fi