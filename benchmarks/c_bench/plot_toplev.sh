#!/usr/bin/env bash

set -euo pipefail

prefix=$1
for model in "NYC_taxi" "mtpl2"; do
  export LLEAVES_BENCHMARK_MODEL=${model}
  pushd build || exit 1
  cmake .. && make

  # high level overview plot
  run_id="${model}_1v_${prefix}"
  toplev.py -l1 -v -I 100 -x, -o "../${run_id}.csv" --no-desc --core C0 taskset -c 0 ./benchmark &&\
   tl-barplot.py --cpu C0 -o "../${run_id}.png" "../${run_id}.csv"

  # detailed level 2 metrics
  run_id="${model}_2v_${prefix}"
  toplev.py -l2 -v -D 500 -o "../${run_id}.txt" --no-desc --core C0 taskset -c 0 ./benchmark

  # detailed level 3 plot
  run_id="${model}_3_${prefix}"
  toplev.py -l3 -I 100 -x, -o "../${run_id}.csv" --no-desc --core C0 taskset -c 0 ./benchmark &&\
   tl-barplot.py --cpu C0 -o "../${run_id}.png" "../${run_id}.csv"

  popd || exit 1
done;