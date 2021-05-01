import lightgbm
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import lleaves
from lleaves.tree_compiler.ast import parse_to_ast

MODEL_DIRS_NUMERICAL = [
    "tests/models/boston_housing/",
    "tests/models/single_tree/",
    "tests/models/tiniest_single_tree/",
]


MODEL_DIRS_CATEGORICAL = [
    "tests/models/mixed_categorical/",
    "tests/models/pure_categorical/",
]


@pytest.fixture(scope="session", params=MODEL_DIRS_NUMERICAL)
def llvm_lgbm_model(request):
    path = request.param
    return (
        lleaves.Model(model_file=path + "model.txt"),
        lightgbm.Booster(model_file=path + "model.txt"),
    )


def test_attribute_similarity(llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    assert llvm_model.num_feature() == lightgbm_model.num_feature()


@pytest.mark.parametrize("model_dir", MODEL_DIRS_NUMERICAL)
@given(data=st.data())
def test_forest_py_mode(data, model_dir):
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    f = parse_to_ast(t_path)

    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=bst.num_feature(),
            min_size=bst.num_feature(),
        )
    )
    assert f._run_pymode(input) == bst.predict([input])


@pytest.mark.parametrize("model_dir", MODEL_DIRS_CATEGORICAL)
@given(data=st.data())
def test_forest_py_mode_cat(data, model_dir):
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    f = parse_to_ast(t_path)

    input = data.draw(
        st.lists(
            st.integers(max_value=20, min_value=-20),
            max_size=bst.num_feature(),
            min_size=bst.num_feature(),
        )
    )
    print(input)
    print(f._run_pymode(input), bst.predict([input]))
    assert f._run_pymode(input) == bst.predict([input])


@settings(deadline=1000)
@given(data=st.data())
def test_forest_llvm_mode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )
    assert llvm_model.predict([input]) == lightgbm_model.predict([input])
