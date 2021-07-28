from ctypes import POINTER, c_double

import numpy as np
from lightgbm import Booster

from lleaves import Model


def test_parallel_iteration():
    llvm_model = Model(model_file="tests/models/NYC_taxi/model.txt")
    lgbm_model = Booster(model_file="tests/models/NYC_taxi/model.txt")
    llvm_model.compile()

    data = np.array(4 * [5 * [1.0]], dtype=np.float64)
    data_flat = np.array(data.reshape(data.size), dtype=np.float64)
    np.testing.assert_almost_equal(
        llvm_model.predict(data, n_jobs=4), lgbm_model.predict(data), decimal=14
    )

    ptr_data = data_flat.ctypes.data_as(POINTER(c_double))
    preds = np.zeros(4, dtype=np.float64)
    ptr_preds = preds.ctypes.data_as(POINTER(c_double))

    llvm_model._c_entry_func(ptr_data, ptr_preds, 2, 4)
    preds_l = list(preds)
    assert preds_l[0] == 0.0 and preds_l[1] == 0.0
    assert preds_l[2] != 0.0 and preds_l[3] != 0.0
    llvm_model._c_entry_func(ptr_data, ptr_preds, 0, 2)
    preds_l = list(preds)
    assert preds_l[0] != 0.0 and preds_l[1] != 0.0
