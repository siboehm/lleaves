import lightgbm as lgb
import numpy.testing as npt

import lleaves


def test_null_prediction(tmp_path):
    model_path = "tests/models/tiniest_single_tree/model.txt"
    llvm_model = lleaves.Model(model_file=model_path)
    lgbm_model = lgb.Booster(model_file=model_path)

    nan = float("NaN")
    inf = float("Inf")
    data = [3 * [nan], 3 * [inf], [nan, inf, nan], [0.0, nan, 0.0], [nan, 0.0, nan]]
    npt.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))
