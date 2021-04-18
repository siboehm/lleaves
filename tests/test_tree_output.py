import json

import lleaves
from lleaves.tree_compiler.decision_tree import Forest
import lightgbm
import pytest
from hypothesis import given, strategies as st

MODEL_DIRS = [
    # "tests/models/boston_housing/",
    ("tests/models/single_tree/", 10),
    ("tests/models/tiniest_single_tree/", 3),
]


@pytest.mark.parametrize("model_dir, n_attributes", MODEL_DIRS)
@given(data=st.data())
def test_forest_py_mode(data, model_dir, n_attributes):
    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=n_attributes,
            min_size=n_attributes,
        )
    )
    j_path = model_dir + "model.json"
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    with open(j_path, "r") as f:
        j = json.load(f)
    f = Forest(j)

    assert f._run_pymode(input) == bst.predict([input])[0]


@pytest.mark.parametrize("model_dir, n_attributes", MODEL_DIRS)
@given(data=st.data())
def test_forest_llvm_mode(data, model_dir, n_attributes):
    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=n_attributes,
            min_size=n_attributes,
        )
    )
    j_path = model_dir + "model.json"
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    lgbm = lleaves.LGBM(file_path=j_path)
    assert lgbm(input) == bst.predict([input])[0]
