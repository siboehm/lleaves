from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.random
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lleaves import Model

numpy.random.seed(1337)


@pytest.fixture(scope="session", params=["pure_cat", "mixed_cat"])
def categorical_model_txt(tmpdir_factory, request):
    n_features = 5
    n_rows = 500
    # specify a tree and generate data from it
    if request.param == "pure_cat":
        n_categorical = 5

        def tree(axis):
            if axis[0] < 5:
                if axis[1] < 5:
                    return 2
                elif axis[0] < 2:
                    return 50
                else:
                    if axis[2] > 7:
                        return 10
                    return 2
            else:
                if axis[1] > 8:
                    return 99
                return 0

    else:
        n_categorical = 3

        def tree(axis):
            if axis[0] < 5:
                if axis[1] < 5:
                    return 2
                elif axis[1] < 2:
                    return 50
                else:
                    if axis[2] > 7:
                        return 100
                    return -32
            else:
                if axis[4] > 0.8:
                    return 99
                else:
                    if axis[3] < 0.2:
                        return -20
                return 0

    train_data_cat = np.random.randint(1, 10, size=(n_rows, n_categorical))
    if n_features - n_categorical > 0:
        train_data_num = np.random.rand(n_rows, n_features - n_categorical)
        train_data = np.concatenate([train_data_cat, train_data_num], axis=1)
    else:
        train_data = train_data_cat

    label = np.apply_along_axis(tree, axis=1, arr=train_data)
    train_data = lgb.Dataset(
        train_data, label=label, categorical_feature=list(range(n_categorical))
    )

    param = {}
    lightgbm_model = lgb.train(
        param, train_data, 1, categorical_feature=list(range(n_categorical))
    )

    tmpdir = tmpdir_factory.mktemp("model")
    model_path = tmpdir / "model.txt"
    lightgbm_model.save_model(str(model_path))
    return model_path


def test_large_categorical(tmpdir_factory):
    # test categorical var with >32 different entries
    def f(x):
        return x % 3

    train_data_cat = np.repeat(
        np.expand_dims(np.arange(1, 150), axis=1), repeats=40, axis=0
    )
    label = np.apply_along_axis(f, axis=1, arr=train_data_cat).flatten()
    train_data = lgb.Dataset(train_data_cat, label=label, categorical_feature=[0])
    lightgbm_model = lgb.train({}, train_data, 1, categorical_feature=[0])

    tmpdir = tmpdir_factory.mktemp("model")
    model_path = str(tmpdir / "model.txt")
    lightgbm_model.save_model(model_path)
    assert "decision_type=9 9 9" in Path(model_path).read_text()

    llvm_model = Model(model_file=model_path)
    llvm_model.compile()
    lgbm_model = lgb.Booster(model_file=model_path)
    tests_data = np.expand_dims(np.arange(0, 210), axis=1)
    numpy.testing.assert_equal(
        llvm_model.predict(tests_data), lgbm_model.predict(tests_data)
    )


def test_predict_pandas_categorical(tmpdir_factory):
    def f(x):
        t = ord(x["C1"]) - ord(x["C2"]) ** 2 + ord(x["C3"]) ** 3 + x["A"]
        return t / 100000

    train_df = pd.DataFrame(
        {
            "C1": 500 * ["a"] + 500 * ["b"] + 500 * ["c"],
            "C2": 500 * ["b"] + 500 * ["c"] + 500 * ["d"],
            "C3": 10 * (50 * ["x"] + 50 * ["y"] + 25 * ["z"] + 25 * ["w"]),
            "A": 5 * (100 * [10] + 100 * [-10] + 100 * [-30]),
        }
    )
    result = train_df.apply(f, axis=1)
    train_df["C1"] = train_df["C1"].astype("category")
    train_df["C2"] = train_df["C2"].astype("category")
    train_df["C3"] = train_df["C3"].astype("category")
    train_data = lgb.Dataset(train_df, label=result, categorical_feature="auto")
    lightgbm_model = lgb.train({}, train_data, 3, categorical_feature="auto")
    assert len(lightgbm_model.pandas_categorical) == 3

    tmpdir = tmpdir_factory.mktemp("model")
    model_path = str(tmpdir / "model.txt")
    lightgbm_model.save_model(model_path)

    llvm_model = Model(model_file=model_path)
    llvm_model.compile()
    lgbm_model = lgb.Booster(model_file=model_path)

    df = pd.DataFrame(
        {
            "C1": 2 * ["a", "b", "c", "c"],
            "C2": 2 * ["b", "c", "d", "b"],
            "C3": 2 * ["x", "y", "z", "w"],
            "A": [10, -10, 100, -30, -30, 10, 1, 100],
        },
    ).astype({"C1": "category", "C2": "category", "C3": "category"})
    df2 = df.copy()
    # reorder categories to ensure that same letter now maps to a different pandas code
    df2["C1"] = df2["C1"].cat.set_categories(["c", "b", "a"])
    df2["C2"] = df2["C2"].cat.set_categories(["a", "z", "w"])
    assert not (list(df["C1"].cat.codes) == list(df2["C1"].cat.codes))
    assert not (list(df["C2"].cat.codes) == list(df2["C2"].cat.codes))

    df3 = pd.DataFrame(
        {
            "C1": 2 * ["a", "b", "c", "s"],
            "C2": 2 * ["b", "c", "v", "b"],
            "C3": 2 * ["x", "u", "z", "w"],
            "A": [10, -10, 100, -30, -30, 10, 1, 100],
        },
    ).astype({"C1": "category", "C2": "category", "C3": "category"})

    # prediction from lightgbm should still be equal
    numpy.testing.assert_equal(lightgbm_model.predict(df), lightgbm_model.predict(df2))
    test_data = [
        df,
        df2,
        df3,
        train_df,
        np.array([[1, 1, 1, 10], [2, -2, 2, 10], [0, 1, 2, 10], [2, 1, 0, -10]]),
    ]
    for data in test_data:
        numpy.testing.assert_equal(llvm_model.predict(data), lgbm_model.predict(data))

    data = (
        pd.DataFrame(
            {
                "C1": 2 * ["b", "a", "a", "a"],
                "C2": 2 * ["d", "c", "a", "b"],
                "C3": 2 * ["z", "w", "z", "y"],
                "A": [10, -10, 100, -30, -30, 10, 1, 100],
            }
        ),
    )
    with pytest.raises(ValueError):
        # no categories
        llvm_model.predict(data)


@settings(deadline=1000)
@given(data=st.data())
def test_categorical_prediction_llvm_real(data, categorical_model_txt):
    lgbm_model = lgb.Booster(model_file=str(categorical_model_txt))
    llvm_model = Model(model_file=categorical_model_txt)
    llvm_model.compile()

    input = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2**31 - 2),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )
    assert llvm_model.predict([input]) == lgbm_model.predict([input])


def test_pure_categorical_prediction():
    llvm_model = Model("tests/models/pure_categorical/model.txt")
    llvm_model.compile()
    lgbm_model = lgb.Booster(model_file="tests/models/pure_categorical/model.txt")

    results = [12.616231057968633, 10.048276920678525, 9.2489478721549396]
    for data, res_idx in zip(
        [
            [0, 9, 0],
            [0, -1, 0],
            [1, 9, 0],
            [0, 6, 5],
            [1, 5, 1],
            [2, 5, 1],
            [4, 5, 1],
            [5, 5, 9],
            [6, 5, 3],
            [9, 5, 2],
            [-1, 5, 2],
        ],
        [0, 1, 0, 0, 1, 1, 1, 2, 2, 2, 2],
    ):
        assert lgbm_model.predict([data]) == [results[res_idx]]
        assert llvm_model.predict([data]) == [results[res_idx]]

    na = float("NaN")
    inf = float("Inf")
    for data, res_idx in zip(
        [
            [na, 9.0, 0.0],
            [na, 0.0, 0.0],
            [inf, 0.0, 0.0],
            [0.0, na, 0.0],
            [4.0, inf, 0.0],
            [na, na, 0.0],
        ],
        [0, 2, 2, 1, 1, 2],
    ):
        assert lgbm_model.predict([data]) == [results[res_idx]]
        assert llvm_model.predict([data]) == [results[res_idx]]
