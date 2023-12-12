## Running the benchmarks

Create a conda environment and activate it:
```commandline
conda env create -f environment.yml
conda activate lleaves
```

Modify [setup.py](../setup.py) to not exclude the `benchmark` package:
```python
packages=find_packages(exclude=[])  # Used to be: exclude=["benchmark"]
```

Install the packages in the environment (optionally, in development mode with `-e`):
```commandline
python -m pip install --no-build-isolation -e .
```

Generate the test data and train the necessary models:
```commandline
./benchmarks/data/setup_data.sh
cd benchmarks
python train_NYC_model.py
```

Finally, run the benchmarks from within the benchmarks folder:
```commandline
python benchmark.py
```
