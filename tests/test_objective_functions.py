import lightgbm as lgb
import numpy as np
import pytest

from lleaves import Model


@pytest.fixture(
    params=[
        "binary sigmoid:10",
        "binary sigmoid:3",
        "binary sigmoid:0.1",
        "cross_entropy",
        "xentropy",
        "xentlambda",
        "cross_entropy_lambda",
        "poisson",
        "gamma",
        "tweedie",
        "regression sqrt",
        "regression",
    ]
)
def modified_model_txt(request, tmp_path):
    model_filep = tmp_path / request.param.replace(" ", "_")
    with open("tests/models/leaf_scan/model.txt") as modelfile, open(
        model_filep, "w"
    ) as tmpfile:
        for line in modelfile:
            if line.startswith("objective="):
                tmpfile.write("objective=" + request.param + "\n")
            else:
                tmpfile.write(line)
    return str(model_filep)


def test_all_obj_funcs(modified_model_txt):
    data = np.expand_dims(np.arange(-5, 5, 0.20), axis=1)
    llvm_model = Model(model_file=modified_model_txt)
    llvm_model.compile()
    lgbm_model = lgb.Booster(model_file=modified_model_txt)
    np.testing.assert_almost_equal(lgbm_model.predict(data), llvm_model.predict(data))


@pytest.mark.parametrize(
    "objective, raw_score",
    [
        ("regression", False),
        ("regression", True),
        ("regression sqrt", False),
        ("regression sqrt", True),
        ("binary", False),
        ("binary", True),
        ("multiclass", False),
        ("multiclass", True),
    ],
)
def test_basic(tmp_path, objective, raw_score):
    X = np.expand_dims(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]), axis=1)
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    train_data = lgb.Dataset(X, label=y, categorical_feature=[0])
    params = {"objective": objective}
    if objective == "multiclass":
        params["num_class"] = 2
    bst = lgb.train(params, train_data, 1, categorical_feature=[0])

    reg_model_f = str(tmp_path / f"{objective}.txt")
    bst.save_model(reg_model_f)
    llvm_model = Model(model_file=reg_model_f)
    llvm_model.compile(raw_score=raw_score)
    np.testing.assert_almost_equal(
        bst.predict(X, raw_score=raw_score), llvm_model.predict(X)
    )
