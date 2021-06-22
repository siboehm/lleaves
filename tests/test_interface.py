import lightgbm as lgb
import numpy as np
import pytest

import lleaves


def test_interface(tmp_path):
    lgbm = lgb.Booster(model_file="tests/models/tiniest_single_tree/model.txt")
    llvm = lleaves.Model("tests/models/tiniest_single_tree/model.txt")
    llvm.compile()

    for arr in [np.array([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]]:
        with pytest.raises(ValueError) as err1:
            llvm.predict(arr)
        with pytest.raises(ValueError) as err2:
            lgbm.predict(arr)

        assert "2 dimensional" in err1.value.args[0]
        assert "2 dimensional" in err2.value.args[0]


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
    llvm = lleaves.Model("tests/models/NYC_taxi/model.txt")
    assert not cachefp.exists()
    llvm.compile(cache=cachefp)
    assert cachefp.exists()
    # we compiled model.txt to IR to asm
    assert llvm._IR_module is not None
    res = llvm.predict([5 * [0.0], 5 * [1.0], 5 * [-1.0]])

    llvm = lleaves.Model("tests/models/NYC_taxi/model.txt")
    llvm.compile(cache=cachefp)
    # we never compiled model.txt to IR
    assert llvm._IR_module is None
    np.testing.assert_equal(res, llvm.predict([5 * [0.0], 5 * [1.0], 5 * [-1.0]]))
