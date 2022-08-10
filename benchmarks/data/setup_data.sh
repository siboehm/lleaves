#!/usr/bin/env bash

set -euox pipefail

pushd benchmarks/data/
wget -q https://f003.backblazeb2.com/file/lleaves-benchmark/benchmark_data.zip
unzip benchmark_data.zip
python gen_npy.py
popd