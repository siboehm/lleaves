# lleaves 🍃
![CI](https://github.com/siboehm/lleaves/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/lleaves/badge/?version=latest)](https://lleaves.readthedocs.io/en/latest/?badge=latest)

A LLVM-based compiler for LightGBM decision trees.

`lleaves` converts trained LightGBM models to optimized machine code, speeding-up inference by up to 10x.

## Example

```python
lgbm_model = lightgbm.Model(model_file="NYC_taxi/model.txt")
%timeit lgbm_model.predict(df)
# 11.6 s ± 442 ms

llvm_model = lleaves.Model(model_file="NYC_taxi/model.txt")
llvm_model.compile()
%timeit llvm_model.predict(df)
# 1.84 s ± 68.7 ms
```

## Why lleaves?
- Speed: Both low-latency single-row prediction and high-throughput batch-prediction.
- Drop-in replacement: The interface of `lleaves.Model` is a subset of `LightGBM.Booster`.
- Dependencies: `llvmlite` and `numpy`. LLVM comes statically linked.

Some LightGBM features are not yet implemented: multiclass prediction, linear models.

## Benchmarks
Ran on Intel Xeon Haswell, 8vCPUs.
Some of the variance is due to performance interference.

Datasets: NYC-taxi (mostly numerical features), Airlines (categorical features with high cardinality)

#### Small batches (single-threaded)
![benchmark small batches](https://raw.githubusercontent.com/siboehm/lleaves/master/benchmarks/1.png)
#### Large batches (multi-threaded)
![benchmark large batches](https://raw.githubusercontent.com/siboehm/lleaves/master/benchmarks/4.png)

## Development
```bash
conda env create
conda activate lleaves
pip install -e .
pre-commit install
pytest
```
