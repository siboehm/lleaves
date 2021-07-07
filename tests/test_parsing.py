from pathlib import Path

import pytest

from lleaves.compiler.ast.scanner import scan_model_file
from lleaves.data_processing import extract_pandas_traintime_categories


def test_parser():
    model_file = "tests/models/boston_housing/model.txt"
    result = scan_model_file(model_file)
    assert result["general_info"]["max_feature_idx"] == 12
    assert len(result["trees"]) == 100

    tree_3 = result["trees"][3]
    assert tree_3["num_leaves"] == 18
    assert tree_3["left_child"] == [
        1,
        3,
        -2,
        -1,
        9,
        8,
        7,
        10,
        15,
        16,
        -5,
        -9,
        -8,
        14,
        -14,
        -6,
        -3,
    ]

    tree_95 = result["trees"][95]
    assert tree_95["Tree"] == 95
    assert tree_95["num_leaves"] == 21

    model_file = "tests/models/tiniest_single_tree/model.txt"
    result = scan_model_file(model_file)
    assert len(result["trees"]) == 1
    tree_0 = result["trees"][0]
    assert tree_0["num_leaves"] == 4


def test_parsing_pandas(tmp_path):
    mod_model_file = tmp_path / "mod_model.txt"
    model_file = Path("tests/models/pure_categorical/model.txt")
    with open(model_file, "r") as file:
        lines = file.readlines()
    assert lines[-1].startswith("pandas_categorical")
    lines[
        -1
    ] = 'pandas_categorical:[["a", "b", "c"], ["b", "c", "d"], ["w", "x", "y", "z"]]'

    with open(mod_model_file, "x") as file:
        file.writelines(lines)

    pandas_categorical = extract_pandas_traintime_categories(model_file)
    assert pandas_categorical is None
    pandas_categorical = extract_pandas_traintime_categories(mod_model_file)
    assert pandas_categorical == [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["w", "x", "y", "z"],
    ]


def test_parsing_pandas_broken_file(tmp_path):
    mod_model_file = tmp_path / "mod_model.txt"
    lines = 100 * ["onelineonly"]
    with open(mod_model_file, "x") as file:
        file.writelines(lines)

    # terminates and raises on garbage file
    with pytest.raises(ValueError):
        _ = extract_pandas_traintime_categories(mod_model_file)
