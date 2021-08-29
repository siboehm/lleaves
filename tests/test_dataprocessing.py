from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lleaves.data_processing import (
    data_to_ndarray,
    extract_n_features_n_classes,
    extract_pandas_traintime_categories,
    ndarray_to_ptr,
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

    res = extract_n_features_n_classes(model_file)
    assert res["n_class"] == 1
    assert res["n_feature"] == 5
    with pytest.raises(ValueError):
        extract_n_features_n_classes(mod_model_file)


def test_no_data_modification():
    # the data shouldn't be modified during conversion
    data = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    for dtype in [np.float32, np.float64, np.int64]:
        orig = np.array(data, dtype=dtype)
        pred = np.array(data, dtype=dtype)

        ndarray_to_ptr(data_to_ndarray(pred))

        np.testing.assert_array_equal(pred, orig)
        assert pred.dtype == orig.dtype

    for dtype in [np.float32, np.float64, np.int64]:
        orig = pd.DataFrame(data).astype(dtype)
        pred = pd.DataFrame(data).astype(dtype)
        ndarray_to_ptr(data_to_ndarray(pred, []))
        pd.testing.assert_frame_equal(pred, orig)

    data = [["a", "b"], ["b", "a"]]
    orig = pd.DataFrame(data).astype("category")
    pred = pd.DataFrame(data).astype("category")
    ndarray_to_ptr(data_to_ndarray(pred, data))
    pd.testing.assert_frame_equal(pred, orig)
