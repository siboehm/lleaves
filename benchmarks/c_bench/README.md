# Setting up Google Benchmark

To specify the model to be benchmarked 
set the environment variable ``LLEAVES_BENCHMARK_MODEL`` to one of
`airline`, `NYC_taxi`, `mtpl2`.

To download the data used for benchmarking there is a bash script `benchmarks/data/setup_data.sh`.

```bash
mkdir build && cd build
export LLEAVES_BENCHMARK_MODEL="mtpl2"
cmake .. && make
./c_bench
```

There is a script to use [toplev](https://github.com/andikleen/pmu-tools) to generate
some plots about CPU bottlenecks.
Make sure `toplev.py` is on your PATH and run:
```bash
./plot_toplev.sh <some prefix>
```