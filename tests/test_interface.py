import lightgbm as lgb
import numpy as np
import pytest

import lleaves


def test_interface(tmp_path):
    lgbm = lgb.Booster(model_file="tests/models/tiniest_single_tree/model.txt")
    llvm = lleaves.Model("tests/models/tiniest_single_tree/model.txt")
    llvm.compile()

    for arr in [
        np.array([1.0, 1.0, 1.0]),
        [1.0, 1.0, 1.0],
    ]:
        with pytest.raises(ValueError) as err1:
            llvm.predict(arr)
        with pytest.raises(ValueError) as err2:
            lgbm.predict(arr)
        assert "dimension" in err1.value.args[0]
        assert "dimension" in err2.value.args[0]

    with pytest.raises(ValueError):
        wrong_shape_2D = (np.array(3 * [(llvm.num_feature() + 1) * [1.0]]),)
        llvm.predict(wrong_shape_2D)


@pytest.mark.parametrize(
    "model_file, n_args",
    [
        ("tests/models/pure_categorical/model.txt", 3),
        ("tests/models/tiniest_single_tree/model.txt", 3),
    ],
)
def test_input_dtypes(model_file, n_args):
    lgbm = lgb.Booster(model_file=model_file)
    llvm = lleaves.Model(model_file)
    llvm.compile()

    arr = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    assert llvm.predict(arr) == lgbm.predict(arr)
    arr = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    assert llvm.predict(arr) == lgbm.predict(arr)
    arr = np.array([[0, 0, 0]], dtype=np.int32)
    assert llvm.predict(arr) == lgbm.predict(arr)
    arr = [[0, 0, 0]]
    assert llvm.predict(arr) == lgbm.predict(arr)


def test_cache_model(tmp_path):
    cachefp = tmp_path / "model.bin"
    pure_cat_llvm = lleaves.Model("tests/models/pure_categorical/model.txt")
    assert not cachefp.exists()
    pure_cat_llvm.compile(cache=cachefp)
    assert cachefp.exists()
    # we compiled model.txt to IR to asm
    res = pure_cat_llvm.predict([3 * [0.0], 3 * [1.0], 3 * [-1.0]])

    cached_model = lleaves.Model("tests/models/tiniest_single_tree/model.txt")
    cached_model.compile(cache=cachefp)

    tiniest_llvm = lleaves.Model("tests/models/tiniest_single_tree/model.txt")
    tiniest_llvm.compile()

    # the cache was loaded (which was different from the model.txt passed)
    np.testing.assert_equal(
        cached_model.predict([3 * [0.0], 3 * [1.0], 3 * [-1.0]]),
        pure_cat_llvm.predict([3 * [0.0], 3 * [1.0], 3 * [-1.0]]),
    )
    # sanity test
    np.testing.assert_equal(
        pure_cat_llvm.predict([3 * [0.0], 3 * [1.0], 3 * [-1.0]]), res
    )
