# Setting up C++ benchmark suite

To specify the model to be benchmarked 
set the environment variable ``LLEAVES_BENCHMARK_MODEL`` to one of
`airline`, `NYC_taxi`, `mtpl2`.

```bash
mkdir build && cd build
export LLEAVES_BENCHMARK_MODEL="mtpl2"
cmake .. && make
./benchmark
```