#!/usr/bin/env bash
set -euox pipefail

export PYTHON_VERSION=$1

./benchmarks/data/setup_data.sh
pytest -v tests

# Check documentation build only in one job, also do releases
if [ "${PYTHON_VERSION}" = "3.9" ]; then
  pushd docs
  make html
  popd

  python -m pip install build
  python -m build --sdist
  python -m build --wheel
fi