from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.random
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lleaves import Model
from lleaves.tree_compiler.ast import parse_to_ast
from lleaves.tree_compiler.utils import bitset_to_py_list

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
        train_data, label=label, categorical_feature=(i for i in range(n_categorical))
    )

    param = {}
    lightgbm_model = lgb.train(param, train_data, 1)

    tmpdir = tmpdir_factory.mktemp("model")
    model_path = tmpdir / "model.txt"
    lightgbm_model.save_model(str(model_path))
    return model_path


def test_large_categorical(tmpdir_factory):
    # test categorical var with >32 different entries
    # will result in decision_type == 9
    f = lambda x: x % 3
    train_data_cat = np.repeat(
        np.expand_dims(np.arange(1, 100), axis=1), repeats=40, axis=0
    )
    label = np.apply_along_axis(f, axis=1, arr=train_data_cat)
    train_data = lgb.Dataset(train_data_cat, label=label, categorical_feature=[0])
    lightgbm_model = lgb.train({}, train_data, 1)

    tmpdir = tmpdir_factory.mktemp("model")
    model_path = str(tmpdir / "model.txt")
    lightgbm_model.save_model(model_path)
    assert "decision_type=9 9 9" in Path(model_path).read_text()

    llvm_model = Model(model_file=model_path)
    lgbm_model = lgb.Booster(model_file=model_path)
    tests_data = np.expand_dims(np.arange(0, 210), axis=1)
    numpy.testing.assert_equal(
        llvm_model.predict(tests_data), lgbm_model.predict(tests_data)
    )


@pytest.mark.parametrize(
    "threshold, result",
    zip(
        [4, 2, 576, 22, 100000],
        [
            [
                2,
            ],
            [
                1,
            ],
            [6, 9],
            [1, 2, 4],
            [5, 7, 9, 10, 15, 16],
        ],
    ),
)
def test_pymode_cat_threshold(threshold, result):
    assert bitset_to_py_list(threshold) == result


@given(data=st.data())
@settings(max_examples=50)
def test_mixed_categorical_prediction_pymode_real(data, categorical_model_txt):
    llvm_model = parse_to_ast(str(categorical_model_txt))
    lgbm_model = lgb.Booster(model_file=str(categorical_model_txt))
    input = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2 ** 31 - 2),
            max_size=lgbm_model.num_feature(),
            min_size=lgbm_model.num_feature(),
        )
    )

    assert llvm_model._run_pymode([input]) == lgbm_model.predict([input])


@given(data=st.data())
def test_categorical_prediction_llvm_real(data, categorical_model_txt):
    lgbm_model = lgb.Booster(model_file=str(categorical_model_txt))
    llvm_model = Model(model_file=categorical_model_txt)

    input = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2 ** 31 - 2),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )
    assert llvm_model.predict([input]) == lgbm_model.predict([input])


def test_pure_categorical_prediction():
    llvm_forest = parse_to_ast("tests/models/pure_categorical/model.txt")
    llvm_model = Model("tests/models/pure_categorical/model.txt")
    lgbm_model = lgb.Booster(model_file="tests/models/pure_categorical/model.txt")

    results = [12.616231057968633, 10.048276920678525, 9.2489478721549396]
    for data, res_idx in zip(
        [
            [0, 9, 0],
            [1, 9, 0],
            [0, 6, 5],
            [1, 5, 1],
            [2, 5, 1],
            [4, 5, 1],
            [5, 5, 9],
            [6, 5, 3],
            [9, 5, 2],
        ],
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
    ):
        assert llvm_forest._run_pymode([data]) == [results[res_idx]]
        assert llvm_model.predict([data]) == [results[res_idx]]
        assert lgbm_model.predict([data]) == [results[res_idx]]
