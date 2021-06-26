import os

import lightgbm
import numpy as np
import pandas as pd
import pytest

import lleaves


@pytest.mark.skipif(
    os.environ.get("CI", "false") == "true",
    reason="We don't want to download the datasets on CI",
)
def test_benchmark_datsets_correct_output():
    from benchmarks.benchmark import NYC_used_columns
    from benchmarks.train_NYC_model import feature_enginering

    model_file_NYC = "tests/models/NYC_taxi/model.txt"
    model_file_airline = "tests/models/airline/model.txt"

    df = pd.read_parquet(
        "benchmarks/data/yellow_tripdata_2016-01.parquet", columns=NYC_used_columns
    )
    NYC_X = feature_enginering().fit_transform(df).astype(np.float32)

    df = pd.read_csv("benchmarks/data/airline_data_factorized.csv")
    airline_X = df.to_numpy(np.float32)

    for model_file, data in [(model_file_NYC, NYC_X), (model_file_airline, airline_X)]:
        lgb = lightgbm.Booster(model_file=model_file)
        llvm = lleaves.Model(model_file=model_file)
        llvm.compile()
        np.testing.assert_almost_equal(
            lgb.predict(data, n_jobs=4), llvm.predict(data, n_jobs=4), decimal=15
        )
