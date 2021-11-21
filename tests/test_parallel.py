from ctypes import POINTER, c_double

import numpy as np


def test_parallel_edgecases(NYC_llvm, NYC_lgbm):
    # single row, multiple threads
    data = np.array(1 * [NYC_lgbm.num_feature() * [1.0]], dtype=np.float64)
    np.testing.assert_almost_equal(
        NYC_llvm.predict(data, n_jobs=4), NYC_lgbm.predict(data), decimal=14
    )

    # last thread has only one prediction (batchsize is ceil(19/7)=3)
    data = np.array(19 * [NYC_lgbm.num_feature() * [1.0]], dtype=np.float64)
    np.testing.assert_almost_equal(
        NYC_llvm.predict(data, n_jobs=7), NYC_lgbm.predict(data), decimal=14
    )


def test_parallel_iteration(NYC_llvm, NYC_lgbm):
    data = np.array(4 * [NYC_lgbm.num_feature() * [1.0]], dtype=np.float64)
    data_flat = np.array(data.reshape(data.size), dtype=np.float64)
    np.testing.assert_almost_equal(
        NYC_llvm.predict(data, n_jobs=4), NYC_lgbm.predict(data), decimal=14
    )

    ptr_data = data_flat.ctypes.data_as(POINTER(c_double))
    preds = np.zeros(4, dtype=np.float64)
    ptr_preds = preds.ctypes.data_as(POINTER(c_double))

    NYC_llvm._c_entry_func(ptr_data, ptr_preds, 2, 4)
    preds_l = list(preds)
    assert preds_l[0] == 0.0 and preds_l[1] == 0.0
    assert preds_l[2] != 0.0 and preds_l[3] != 0.0
    NYC_llvm._c_entry_func(ptr_data, ptr_preds, 0, 2)
    preds_l = list(preds)
    assert preds_l[0] != 0.0 and preds_l[1] != 0.0
