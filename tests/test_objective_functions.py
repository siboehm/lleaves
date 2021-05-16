from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from lleaves import Model


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_basic(tmp_path, objective):
    X = np.expand_dims(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]), axis=1)
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    train_data = lgb.Dataset(X, label=y, categorical_feature=[0])
    bst = lgb.train({"objective": objective}, train_data, 1)

    reg_model_f = str(tmp_path / f"{objective}.txt")
    bst.save_model(reg_model_f)
    llvm_model = Model(model_file=reg_model_f)
    print(Path(reg_model_f).read_text())
    np.testing.assert_almost_equal(bst.predict(X), llvm_model.predict(X))
