import lightgbm as lgb
import numpy as np
import numpy.random
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lleaves import Model

numpy.random.seed(1337)


@pytest.fixture(scope="session")
def categorical_model_txt(tmpdir_factory):
    n_features = 3
    n_rows = 500
    train_data = np.random.randint(1, 10, size=(n_rows, n_features))

    def tree(axis):
        if axis[0] < 5:
            if axis[1] < 5:
                return 2
            elif axis[0] < 2:
                return 50
            else:
                if axis[2] > 7:
                    return 10
                return 2
        else:
            if axis[1] > 8:
                return 99
            return 0

    label = np.apply_along_axis(tree, axis=1, arr=train_data)
    train_data = lgb.Dataset(
        train_data, label=label, categorical_feature=(i for i in range(n_features))
    )

    param = {}
    lightgbm_model = lgb.train(param, train_data, 1)

    tmpdir = tmpdir_factory.mktemp("model")
    model_path = tmpdir / "model.txt"
    lightgbm_model.save_model(str(model_path))
    return model_path


@given(data=st.data())
def test_categorical_prediction(data, categorical_model_txt):
    print(categorical_model_txt.read_text("utf-8"))

    lgbm_model = lgb.Booster(model_file=str(categorical_model_txt))
    llvm_model = Model(model_file=categorical_model_txt)

    input = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )

    print(llvm_model.predict([input]))
    assert llvm_model.predict([input]) == lgbm_model.predict([input])
