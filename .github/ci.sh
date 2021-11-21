#!/usr/bin/env bash
set -euox pipefail

export PYTHON_VERSION=$1

python -m pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .
./benchmarks/data/setup_data.sh
pytest -v tests

# Check documentation build only in one job, also do releases
if [ "${PYTHON_VERSION}" = "3.7" ]; then
  pushd docs
  make html
  popd

  python -m pip install build
  python -m build --sdist
  python -m build --wheel
fi