import lightgbm as lgb
import numpy.testing as npt

import lleaves


def test_null_prediction_numerical():
    model_path = "tests/models/tiniest_single_tree/model.txt"
    llvm_model = lleaves.Model(model_file=model_path)
    lgbm_model = lgb.Booster(model_file=model_path)

    nan = float("NaN")
    inf = float("Inf")
    data = [
        3 * [nan],
        3 * [inf],
        [nan, inf, nan],
        [0.0, nan, 0.0],
        [0.0, inf, 0.0],
        [0.0, nan, 1.0],
        [0.0, inf, 1.0],
        [0.0, 0.0, nan],
        [0.0, 0.0, inf],
        [0.0, 1.0, nan],
        [0.0, 1.0, inf],
    ]
    npt.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))


def test_null_prediction_categorical():
    model_path = "tests/models/pure_categorical/model.txt"
    llvm_model = lleaves.Model(model_file=model_path)
    lgbm_model = lgb.Booster(model_file=model_path)

    nan = float("NaN")
    inf = float("Inf")
    data = [
        3 * [nan],
        3 * [inf],
        [nan, inf, nan],
        # run both branches with Nans
        [1.0, nan, 0.0],
        [1.0, inf, 0.0],
        [0.0, nan, 0.0],
        [0.0, inf, 0.0],
        [nan, 6.0, 0.0],
        [inf, 6.0, 0.0],
        [nan, 0.0, 0.0],
        [inf, 0.0, 0.0],
    ]
    npt.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))
