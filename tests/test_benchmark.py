import lightgbm
import numpy as np
import pandas as pd

import lleaves
from benchmarks.benchmark import NYC_used_columns
from benchmarks.train_NYC_model import feature_enginering


def test_benchmarks_correct_output():
    model_file_NYC = "tests/models/NYC_taxi/model.txt"
    model_file_airline = "tests/models/airline/model.txt"
    model_file_mtpl2 = "tests/models/mtpl2/model.txt"

    df = pd.read_parquet(
        "benchmarks/data/yellow_tripdata_2016-01.parquet", columns=NYC_used_columns
    )
    NYC_X = feature_enginering().fit_transform(df).astype(np.float64)

    df = pd.read_csv("benchmarks/data/airline_data_factorized.csv")
    airline_X = df.to_numpy(np.float64)

    df = pd.read_parquet("benchmarks/data/mtpl2.parquet")
    mtpl2_X = df.to_numpy(np.float64)

    for model_file, data in [
        (model_file_mtpl2, mtpl2_X),
        (model_file_NYC, NYC_X),
        (model_file_airline, airline_X),
    ]:
        lgb = lightgbm.Booster(model_file=model_file)
        llvm = lleaves.Model(model_file=model_file)
        llvm.compile()
        np.testing.assert_almost_equal(
            lgb.predict(data, n_jobs=2), llvm.predict(data, n_jobs=2), decimal=10
        )


def test_NYC_benchmark_correct_output_sp():
    model_file = "tests/models/NYC_taxi/model.txt"

    df = pd.read_parquet(
        "benchmarks/data/yellow_tripdata_2016-01.parquet", columns=NYC_used_columns
    )
    data = feature_enginering().fit_transform(df).astype(np.float32)

    lgb = lightgbm.Booster(model_file=model_file)
    llvm = lleaves.Model(model_file=model_file)
    llvm.compile(use_fp64=False)
    lgbm_pred = lgb.predict(data, n_jobs=2).astype(np.float32)
    llvm_pred = llvm.predict(data, n_jobs=2)
    assert llvm_pred.dtype == np.float32
    np.testing.assert_allclose(lgbm_pred, llvm_pred, rtol=0.02)
