#include "c_bench.h"
#include <cnpy.h>
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

static void bm_lleaves(benchmark::State &state)
{
  char *model_name = std::getenv("LLEAVES_BENCHMARK_MODEL");

  std::ostringstream model_stream;
  model_stream << "../../data/" << model_name << ".npy";
  std::string model_file = model_stream.str();
  cnpy::NpyArray arr = cnpy::npy_load(model_file);

  auto *loaded_data = arr.data<double>();
  ulong n_preds = arr.shape[0];
  auto *out = (double *)(malloc(n_preds * sizeof(double)));

  for (auto _ : state)
  {
    // predict over the whole input array
    forest_root(loaded_data, out, (int)0, (int)n_preds);
  }
}

BENCHMARK(bm_lleaves)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();