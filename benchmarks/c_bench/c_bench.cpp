#include "c_bench.h"
#include "cnpy.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

#define N_REPEAT 20

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  char *model_name = std::getenv("LLEAVES_BENCHMARK_MODEL");
  std::cout << "Running model " << model_name << "\n";

  std::ostringstream model_stream;
  model_stream << "../../data/" << model_name << ".npy";
  std::string model_file = model_stream.str();
  cnpy::NpyArray arr = cnpy::npy_load(model_file);

  std::cout << "Batchsize: " << arr.shape[0] << "\n";

  auto *loaded_data = arr.data<double>();
  ulong n_preds = arr.shape[0] / (ulong)6;
  auto *out = (double *)(malloc(n_preds * sizeof(double)));

  std::array<double, N_REPEAT> timings{};
  clock_t start, end;
  std::cout << "starting...\n";
  for (size_t i = 0; i < N_REPEAT; ++i) {
    start = clock();
    forest_root(loaded_data, out, (int)0, (int)n_preds);
    end = clock();

    timings[i] = (double)(end - start) / CLOCKS_PER_SEC;
  }
  std::cout << "...ending, took "
            << std::accumulate(timings.begin(), timings.end(), 0.0) << "\n";

  std::cout << "Runtime: " << *std::min_element(timings.begin(), timings.end())
            << "\n";
}
