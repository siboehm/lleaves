import pathlib

import lightgbm as lgb
import numpy.testing as npt
import pytest

import lleaves
from lleaves.tree_compiler.utils import MissingType


# 2: MissingType None, default left
# 4: MissingType 0, default right
# 6: MissingType 0, default left
# 8: MissingType NaN, default left
# 10: MissingType NaN, default right
@pytest.mark.parametrize("decision_type", [2, 4, 6, 8, 10])
def test_zero_as_missing_numerical(tmp_path, decision_type):
    model_txt = tmp_path / "model.txt"
    with open("tests/models/tiniest_single_tree/model.txt") as infile, open(
        model_txt, "w"
    ) as outfile:
        for line in infile.readlines():
            if line.startswith("decision_type"):
                # change missing type from None to Zero
                outfile.write(line.replace("2", str(decision_type)))
            else:
                outfile.write(line)

    lgbm_model = lgb.Booster(model_file=str(model_txt))
    llvm_model = lleaves.Model(model_file=str(model_txt))

    nan = float("NaN")
    data = [
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.5],
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 0.0],
        [-0.01, -0.01, -0.01],
        [0.0, 0.0, 0.0],
        [0.01, 0.01, 0.01],
        [nan, nan, nan],
        [None, None, None],
    ]
    npt.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))


def test_nan_prediction_numerical():
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


def test_nan_prediction_categorical():
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
