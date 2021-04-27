import lightgbm
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import lleaves
from lleaves.tree_compiler.ast import parse_to_forest

MODEL_DIRS = [
    ("tests/models/boston_housing/", 13),
    ("tests/models/single_tree/", 10),
    ("tests/models/tiniest_single_tree/", 3),
]


@pytest.fixture(scope="session", params=MODEL_DIRS, ids=[x[0] for x in MODEL_DIRS])
def model(request):
    path = request.param[0]
    n_attr = request.param[1]
    return (
        lleaves.Model(model_file=path + "model.txt"),
        lightgbm.Booster(model_file=path + "model.txt"),
        n_attr,
    )


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
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    f = parse_to_forest(t_path)

    assert f._run_pymode(input) == bst.predict([input])[0]


@settings(deadline=1000)
@given(data=st.data())
def test_forest_llvm_mode(data, model):
    llvm_model, lightgbm_model, n_attributes = model
    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=n_attributes,
            min_size=n_attributes,
        )
    )
    assert llvm_model.predict([input])[0] == lightgbm_model.predict([input])[0]
