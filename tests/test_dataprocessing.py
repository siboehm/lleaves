from pathlib import Path

import pytest

from lleaves.data_processing import (
    extract_num_feature,
    extract_pandas_traintime_categories,
)


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


def test_n_args_extract(tmp_path):
    mod_model_file = tmp_path / "mod_model.txt"
    model_file = Path("tests/models/mixed_categorical/model.txt")
    with open(model_file, "r") as file:
        lines = file.readlines()

    with open(mod_model_file, "x") as file:
        file.writelines(
            (line for line in lines if not line.startswith("max_feature_idx"))
        )

    assert extract_num_feature(model_file) == 5
    with pytest.raises(ValueError):
        extract_num_feature(mod_model_file)
