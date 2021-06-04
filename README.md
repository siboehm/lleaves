# LLeaves 🐉
A LLVM-based compiler for LightGBM decision trees.

Ingests `model.txt` files from trained LightGBM Models and
converts them into optimized machine code.

## Why LLeaves?
- Drop-in replacement for LightGBM: The interface is a subset of `LightGBM.Booster`.
- Speed: Up to 10x performance increase compared to LightGBM.
- Just two dependencies: `llvmlite` and `numpy`. LLVM comes statically linked.
 
## Why not LLeaves?
Some LightGBM features are not yet implemented in LLeaVes:
- Multiclass prediction
- Multithreading
- Linear Models

## Benchmarks
[benchmark script](benchmarks/benchmark_small_batches.py).
LLeaVes has no support for MT so far and is running single-threaded mode only.
#### NYC-taxi
Numerical features only
![img](benchmarks/NYC_taxi.png)
#### Airlines dataset
Predominantly categorical features with high cardinality
![img](benchmarks/airline.png)

## Development
```bash
conda env create
conda activate lleaves
pip install -e .
pre-commit install
pytest
```

### Tasks
- Come up with a better name (has to be available on PyPI and conda)
- Release GIL and implement multithreading
- Refactor `nodes.py` to split AST-traversal from IR Codegen.
- Experiment with more efficient bitvector storage for categoricals (Int64 instead of Int32).
- Implement final output transformation function in IR instead of numpy ufunc.
- Parse `internal_count` from model.txt, use it for compiler branch prediction hints. 
  (Caveat: Treelite has branch prediction hints and it doesn't with speed at all)
