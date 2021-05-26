# LLeaVes 🐉
A LLVM-based compiler for LightGBM decision trees.

`lleaves` will ingests the `model.txt`-files from your trained LightGBM Models and
converts them to optimized machine code via LLVM IR.

## Why LLeaVes?
- Easy to use: The interface of `lleaves.Model` is a subset of `LightGBM.Booster`.
- Speed: 100x performance increase for single-threaded prediction
- Only two dependencies: `llvmlite` and `numpy`. No C/C++ compiler necessary.
  
## Why not LLeaVes?
Some LightGBM features that are not yet implemented in LLeaVes:
- Multiclass prediction
- Multithreading
- Linear Models

If you'd like them to become part of this library, vote for them on
the [issues page](https://github.com/siboehm/LLeaVes/issues).

## Development
```bash
conda env create
conda activate lleaves
pre-commit install
pytest
```
