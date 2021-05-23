import lightgbm
import numpy as np
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
CAT_BITVEC_CATEGORICAL = [(True, True, True, False, False), (True, True, True)]


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
@settings(max_examples=50)
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
    assert f._run_pymode([input]) == bst.predict([input])


@pytest.mark.parametrize(
    "model_dir, cat_bitvec", zip(MODEL_DIRS_CATEGORICAL, CAT_BITVEC_CATEGORICAL)
)
@settings(max_examples=50)
@given(data=st.data())
def test_forest_py_mode_cat(data, model_dir, cat_bitvec):
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    f = parse_to_ast(t_path)

    input_cats = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2 ** 31 - 2),
            min_size=sum(cat_bitvec),
            max_size=sum(cat_bitvec),
        )
    )
    input_floats = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=bst.num_feature() - sum(cat_bitvec),
            min_size=bst.num_feature() - sum(cat_bitvec),
        )
    )
    input_data = [
        input_cats.pop() if is_cat else input_floats.pop() for is_cat in cat_bitvec
    ]
    assert f._run_pymode([input_data]) == bst.predict([input_data])


@settings(deadline=1000)
@given(data=st.data())
def test_forest_llvm_mode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input_data = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )
    input_data = np.array([input_data])
    assert llvm_model.predict(input_data) == lightgbm_model.predict(input_data)


@settings(max_examples=10)
@given(data=st.data())
def test_batchmode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input_data = []
    for i in range(10):
        input_data.append(
            data.draw(
                st.lists(
                    st.floats(allow_nan=True, allow_infinity=True),
                    max_size=llvm_model.num_feature(),
                    min_size=llvm_model.num_feature(),
                )
            )
        )
    input_data = np.array(input_data)
    np.testing.assert_array_equal(
        llvm_model.predict(input_data), lightgbm_model.predict(input_data)
    )


@pytest.mark.parametrize(
    "model_dir, cat_bitvec", zip(MODEL_DIRS_CATEGORICAL, CAT_BITVEC_CATEGORICAL)
)
@given(data=st.data())
def test_forest_llvm_mode_cat(data, model_dir, cat_bitvec):
    t_path = model_dir + "model.txt"
    lgbm_model = lightgbm.Booster(model_file=t_path)
    llvm_model = lleaves.Model(t_path)

    input_cats = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2 ** 31 - 2),
            min_size=sum(cat_bitvec),
            max_size=sum(cat_bitvec),
        )
    )
    input_floats = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=lgbm_model.num_feature() - sum(cat_bitvec),
            min_size=lgbm_model.num_feature() - sum(cat_bitvec),
        )
    )
    input_data = [
        input_cats.pop() if is_cat else input_floats.pop() for is_cat in cat_bitvec
    ]
    assert llvm_model.predict([input_data]) == lgbm_model.predict([input_data])
