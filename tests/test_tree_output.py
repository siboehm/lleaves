import lightgbm
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import lleaves
from lleaves.tree_compiler.ast import parse_to_forest

MODEL_DIRS = [
    "tests/models/boston_housing/",
    "tests/models/single_tree/",
    "tests/models/tiniest_single_tree/",
]


@pytest.fixture(scope="session", params=MODEL_DIRS)
def llvm_lgbm_model(request):
    path = request.param
    return (
        lleaves.Model(model_file=path + "model.txt"),
        lightgbm.Booster(model_file=path + "model.txt"),
    )


@pytest.mark.parametrize("model_dir", MODEL_DIRS)
@given(data=st.data())
def test_forest_py_mode(data, model_dir):
    t_path = model_dir + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    f = parse_to_forest(t_path)

    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=bst.num_feature(),
            min_size=bst.num_feature(),
        )
    )

    assert f._run_pymode(input) == bst.predict([input])


@settings(deadline=1000)
@given(data=st.data())
def test_forest_llvm_mode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=llvm_model.num_features(),
            min_size=llvm_model.num_features(),
        )
    )
    assert llvm_model.predict([input]) == lightgbm_model.predict([input])
