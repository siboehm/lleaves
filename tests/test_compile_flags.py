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


# we don't test the default, which is 34
@pytest.mark.parametrize("blocksize", [1, 100])
def test_cache_blocksize(blocksize, NYC_data):
    # TODO there should be a test here to make sure 100,actually disables instr blocking
    llvm_model = Model(model_file="tests/models/NYC_taxi/model.txt")
    lgbm_model = Booster(model_file="tests/models/NYC_taxi/model.txt")
    llvm_model.compile(fblocksize=blocksize)

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
