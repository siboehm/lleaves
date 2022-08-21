# lleaves 🍃
![CI](https://github.com/siboehm/lleaves/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/lleaves/badge/?version=latest)](https://lleaves.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/lleaves)](https://pepy.tech/project/lleaves)

A LLVM-based compiler for LightGBM decision trees.

`lleaves` converts trained LightGBM models to optimized machine code, speeding-up prediction by ≥10x.

## Example

```python
lgbm_model = lightgbm.Booster(model_file="NYC_taxi/model.txt")
%timeit lgbm_model.predict(df)
# 12.77s

llvm_model = lleaves.Model(model_file="NYC_taxi/model.txt")
llvm_model.compile()
%timeit llvm_model.predict(df)
# 0.90s 
```

## Why lleaves?
- Speed: Both low-latency single-row prediction and high-throughput batch-prediction.
- Drop-in replacement: The interface of `lleaves.Model` is a subset of `LightGBM.Booster`.
- Dependencies: `llvmlite` and `numpy`. LLVM comes statically linked.

## Installation
`conda install -c conda-forge lleaves` or `pip install lleaves` (Linux and MacOS only).

## Benchmarks
Ran on a dedicated Intel i7-4770 Haswell, 4 cores.
Stated runtime is the minimum over 20.000 runs.

### Dataset: [NYC-taxi](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
mostly numerical features.
|batchsize   | 1  | 10| 100 |
|---|---:|---:|---:|
|LightGBM   | 52.31μs   | 84.46μs   | 441.15μs |
|ONNX  Runtime| 11.00μs | 36.74μs | 190.87μs  |
|Treelite   | 28.03μs   | 40.81μs   | 94.14μs  |
|``lleaves``   | 9.61μs | 14.06μs | 31.88μs  |

### Dataset: [MTPL2](https://www.openml.org/d/41214)
mix of categorical and numerical features.
|batchsize   | 10,000  | 100,000  | 678,000 |
|---|---:|---:|---:|
|LightGBM   | 95.14ms | 992.47ms   | 7034.65ms  |
|ONNX  Runtime | 38.83ms  | 381.40ms  | 2849.42ms  |
|Treelite   | 38.15ms | 414.15ms  | 2854.10ms  |
|``lleaves``  | 5.90ms  | 56.96ms | 388.88ms |

## Advanced Usage
To avoid expensive recompilation, you can call `lleaves.Model.compile()` and pass a `cache=<filepath>` argument.
This will store an ELF (Linux) / Mach-O (macOS) file at the given path when the method is first called.
Subsequent calls of `compile(cache=<same filepath>)` will skip compilation and load the stored binary file instead.
For more info, see [docs](https://lleaves.readthedocs.io/en/latest/).

To eliminate any Python overhead during inference you can link against this generated binary.
For an example of how to do this see `benchmarks/c_bench/`.
The function signature might change between major versions.

## Development
```bash
conda env create
conda activate lleaves
pip install -e .
pre-commit install
./benchmarks/data/setup_data.sh
pytest
```
