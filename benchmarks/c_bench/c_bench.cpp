#include "c_bench.h"
#include <cnpy.h>
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

static void prediction_batch(benchmark::State &state) {
    char *model_name = std::getenv("LLEAVES_BENCHMARK_MODEL");

    std::ostringstream model_stream;
    model_stream << "../../data/" << model_name << ".npy";
    std::string model_file = model_stream.str();
    cnpy::NpyArray arr = cnpy::npy_load(model_file);

    auto *loaded_data = arr.data<double>();
    ulong n_preds = arr.shape[0];
    auto *out = (double *) (malloc(n_preds * sizeof(double)));

    for (auto _: state) {
        // batch-prediction over the whole input array
        forest_root(loaded_data, out, (int) 0, (int) n_preds);
        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(prediction_batch);

static void prediction_single_row(benchmark::State &state) {
    char *model_name = std::getenv("LLEAVES_BENCHMARK_MODEL");

    std::ostringstream model_stream;
    model_stream << "../../data/" << model_name << ".npy";
    std::string model_file = model_stream.str();
    cnpy::NpyArray arr = cnpy::npy_load(model_file);

    auto *loaded_data = arr.data<double>();
    int n_preds = (int) arr.shape[0];
    auto *out = (double *) (malloc(n_preds * sizeof(double)));
    int i = 0;
    for (auto _: state) {
        // batch-prediction over the whole input array
        forest_root(loaded_data, out, 0, 1);
        benchmark::DoNotOptimize(out);
        i = (i + 1) % n_preds;
    }
}

BENCHMARK(prediction_single_row);
BENCHMARK_MAIN();