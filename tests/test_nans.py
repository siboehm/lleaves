import lightgbm as lgb
import numpy as np
import numpy.testing as npt
import pytest

import lleaves


# 2: MissingType None, default left
# 4: MissingType 0, default right
# 6: MissingType 0, default left
# 8: MissingType NaN, default left
# 10: MissingType NaN, default right
@pytest.mark.parametrize(
    "decision_type, threshold_le_zero",
    [
        (0, True),
        (2, True),
        (4, True),
        (6, True),
        (8, True),
        (10, True),
        (0, False),
        (2, False),
        (4, False),
        (6, False),
        (8, False),
        (10, False),
    ],
)
def test_zero_as_missing_numerical(tmp_path, decision_type, threshold_le_zero):
    model_txt = tmp_path / "model.txt"
    with open("tests/models/tiniest_single_tree/model.txt") as infile, open(
        model_txt, "w"
    ) as outfile:
        for line in infile.readlines():
            if line.startswith("decision_type="):
                outfile.write(line.replace("2", str(decision_type)))
            elif threshold_le_zero and line.startswith("threshold="):
                outfile.write(line.replace("0.", "-0."))
            else:
                outfile.write(line)

    lgbm_model = lgb.Booster(model_file=str(model_txt))
    llvm_model = lleaves.Model(model_file=str(model_txt))
    llvm_model.compile()

    nan = float("NaN")
    data = [
        [0.0, 1.0, 1.0],
        [0.0, -1.0, -1.0],
        [0.0, 1.0, 0.5],
        [0.0, -1.0, -0.5],
        [0.0, 0.5, 1.0],
        [0.0, -0.5, -1.0],
        [0.0, 0.5, 0.0],
        [0.0, -0.5, 0.0],
        [-0.01, -0.01, -0.01],
        [0.0, 0.0, 0.0],
        [0.01, 0.01, 0.01],
        [nan, nan, nan],
        [None, None, None],
    ]
    npt.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))


@pytest.mark.parametrize(
    "decision_type, zero_in_bitvec",
    [
        (1, True),
        (3, True),
        (5, True),
        (7, True),
        (9, True),
        (11, True),
        (1, False),
        (3, False),
        (5, False),
        (7, False),
        (9, False),
        (11, False),
    ],
)
def test_zero_as_missing_categorical(tmp_path, decision_type, zero_in_bitvec):
    model_txt = tmp_path / "model.txt"
    with open("tests/models/pure_categorical/model.txt") as infile, open(
        model_txt, "w"
    ) as outfile:
        for line in infile.readlines():
            if line.startswith("decision_type"):
                outfile.write(line.replace("1", str(decision_type)))
            elif line.startswith("cat_threshold") and not zero_in_bitvec:
                outfile.write(line.replace("23", "22"))
            else:
                outfile.write(line)

    lgbm_model = lgb.Booster(model_file=str(model_txt))
    llvm_model = lleaves.Model(model_file=str(model_txt))
    llvm_model.compile()

    nan = float("NaN")
    data = [
        [1.0, 6.0, 0.0],
        [nan, 6.0, 0.0],
        [nan, 1.0, 0.0],
        [None, 1.0, 0.0],
        [1.0, nan, 0.0],
        [3.0, nan, 0.0],
        [3.0, None, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, -0.001, 0.0],
        [1.0, 0.001, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, -0.001, 0.0],
        [3.0, 0.001, 0.0],
        [nan, nan, nan],
        [None, None, None],
    ]
    npt.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))


def test_lightgbm_nan_pred_inconsistency(tmp_path):
    # see https://github.com/dmlc/treelite/issues/277
    model_file = str(tmp_path / "model.txt")
    X = np.array(30 * [[1]] + 30 * [[2]] + 30 * [[0]])
    y = np.array(60 * [5] + 30 * [10])
    train_data = lgb.Dataset(X, label=y, categorical_feature=[0])
    bst = lgb.train({}, train_data, 1, categorical_feature=[0])
    bst.save_model(model_file)

    # just to make sure it's not due to LightGBM model export
    lgbm_model = lgb.Booster(model_file=model_file)
    llvm_model = lleaves.Model(model_file=model_file)
    llvm_model.compile()

    data = np.array([[np.NaN], [0.0], [-0.1], [0.1], [10.0], [np.Inf], [-np.NaN]])
    npt.assert_equal(lgbm_model.predict(data), llvm_model.predict(data))


def test_nan_prediction_numerical():
    model_path = "tests/models/tiniest_single_tree/model.txt"
    llvm_model = lleaves.Model(model_file=model_path)
    llvm_model.compile()
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
    llvm_model.compile()
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
