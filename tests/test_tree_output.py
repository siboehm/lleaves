import json

from lleaves.tree_compiler.decision_tree import Forest
import lightgbm
from hypothesis import given, strategies as st


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=10, min_size=10)
)
def test_forest_py_mode(input):
    path = "tests/models/single_tree/"
    j_path = path + "model.json"
    t_path = path + "model.txt"
    bst = lightgbm.Booster(model_file=t_path)

    with open(j_path, "r") as f:
        j = json.load(f)
    f = Forest(j)

    assert f._run_pymode(input) == bst.predict([input])[0]
