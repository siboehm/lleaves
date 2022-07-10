import io
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from lightgbm import Booster

from benchmarks.benchmark import NYC_used_columns
from benchmarks.train_NYC_model import feature_enginering
from lleaves import Model


@pytest.fixture(scope="session")
def NYC_data():
    df = pd.read_parquet(
        "benchmarks/data/yellow_tripdata_2016-01.parquet", columns=NYC_used_columns
    )
    return feature_enginering().fit_transform(df).astype(np.float64)


@pytest.mark.parametrize("blocksize", [1, 34, 100])
def test_cache_blocksize(blocksize, NYC_data):
    llvm_model = Model(model_file="tests/models/NYC_taxi/model.txt")
    lgbm_model = Booster(model_file="tests/models/NYC_taxi/model.txt")

    os.environ["LLEAVES_PRINT_UNOPTIMIZED_IR"] = "1"
    f = io.StringIO()
    with redirect_stdout(f):
        llvm_model.compile(fblocksize=blocksize)
    os.environ["LLEAVES_PRINT_UNOPTIMIZED_IR"] = "0"

    stdout = f.getvalue()
    # each cache block has an IR block called "instr-block-setup"
    assert "instr-block-setup:" in stdout
    if blocksize == 1:
        assert "instr-block-setup.1:" in stdout
        assert "instr-block-setup.99:" in stdout
        assert "instr-block-setup.100:" not in stdout
    if blocksize == 34:
        # NYC_taxi has 100 trees, hence blocksize 34 should create 3 blocks
        assert "instr-block-setup.1:" in stdout
        assert "instr-block-setup.2:" in stdout
        assert "instr-block-setup.3:" not in stdout
        assert "instr-block-setup.4:" not in stdout
    if blocksize == 100:
        assert "instr-block-setup.1:" not in stdout
        assert "instr-block-setup.2:" not in stdout

    np.testing.assert_almost_equal(
        llvm_model.predict(NYC_data[:1000], n_jobs=2),
        lgbm_model.predict(NYC_data[:1000], n_jobs=2),
    )


def test_small_codemodel(NYC_data):
    llvm_model = Model(model_file="tests/models/NYC_taxi/model.txt")
    lgbm_model = Booster(model_file="tests/models/NYC_taxi/model.txt")
    llvm_model.compile(fcodemodel="small")

    np.testing.assert_almost_equal(
        llvm_model.predict(NYC_data[:1000], n_jobs=2),
        lgbm_model.predict(NYC_data[:1000], n_jobs=2),
    )


def test_no_inline(NYC_data):
    llvm_model = Model(model_file="tests/models/NYC_taxi/model.txt")
    lgbm_model = Booster(model_file="tests/models/NYC_taxi/model.txt")
    llvm_model.compile(finline=False)

    np.testing.assert_almost_equal(
        llvm_model.predict(NYC_data[:1000], n_jobs=2),
        lgbm_model.predict(NYC_data[:1000], n_jobs=2),
    )


def test_function_name():
    llvm_model = Model(model_file="tests/models/tiniest_single_tree/model.txt")
    lgbm_model = Booster(model_file="tests/models/tiniest_single_tree/model.txt")
    llvm_model.compile(froot_func_name="tiniest_single_tree_123132_XXX-")

    data = [
        [1.0] * 3,
        [0.0] * 3,
        [-1.0] * 3,
    ]
    np.testing.assert_almost_equal(
        llvm_model.predict(data, n_jobs=2),
        lgbm_model.predict(data, n_jobs=2),
    )
