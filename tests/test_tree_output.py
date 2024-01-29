import lightgbm
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.datasets import make_blobs, make_classification, make_regression

import lleaves

MODEL_DIRS_NUMERICAL = [
    "tests/models/boston_housing/",
    "tests/models/NYC_taxi/",
    "tests/models/single_tree/",
    "tests/models/tiniest_single_tree/",
    "tests/models/multiclass/",
]


MODEL_DIRS_CATEGORICAL = [
    "tests/models/mixed_categorical/",
    "tests/models/pure_categorical/",
    "tests/models/airline/",
]
CAT_BITVEC_CATEGORICAL = [
    (True, True, True, False, False),
    (True, True, True),
    (True, True, True, True, False, False),
]


@pytest.fixture(scope="session", params=MODEL_DIRS_NUMERICAL)
def llvm_lgbm_model(request):
    path = request.param
    llvm = lleaves.Model(model_file=path + "model.txt")
    llvm.compile()
    return (
        llvm,
        lightgbm.Booster(model_file=path + "model.txt"),
    )


@pytest.fixture(scope="session", params=MODEL_DIRS_NUMERICAL)
def llvm_lgbm_model_single_precision(request):
    path = request.param
    llvm = lleaves.Model(model_file=path + "model.txt")
    llvm.compile(use_fp64=False)
    return (
        llvm,
        lightgbm.Booster(model_file=path + "model.txt"),
    )


@pytest.fixture(
    scope="session", params=zip(MODEL_DIRS_CATEGORICAL, CAT_BITVEC_CATEGORICAL)
)
def llvm_lgbm_model_cat(request):
    path, bitvec = request.param

    llvm = lleaves.Model(model_file=path + "model.txt")
    llvm.compile()
    return (
        llvm,
        lightgbm.Booster(model_file=path + "model.txt"),
        bitvec,
    )


def test_attribute_similarity(llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    assert llvm_model.num_feature() == lightgbm_model.num_feature()
    assert (
        llvm_model.num_model_per_iteration() == lightgbm_model.num_model_per_iteration()
    )
    assert llvm_model.num_trees() == lightgbm_model.num_trees()


@settings(deadline=1000)
@given(data=st.data())
def test_forest_llvm_mode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input_data = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )
    input_data = np.array([input_data])
    np.testing.assert_array_almost_equal(
        llvm_model.predict(input_data), lightgbm_model.predict(input_data)
    )


@settings(max_examples=10)
@given(data=st.data())
def test_batchmode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input_data = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=20 * llvm_model.num_feature(),
            min_size=20 * llvm_model.num_feature(),
        )
    )
    input_data = np.array(input_data).reshape((20, llvm_model.num_feature()))

    lgbm_result = lightgbm_model.predict(input_data)
    llvm_result = llvm_model.predict(input_data)
    assert llvm_result.dtype == np.float64
    np.testing.assert_almost_equal(lgbm_result, llvm_result)


@settings(max_examples=10)
@given(data=st.data())
def test_batchmode_single_precision(data, llvm_lgbm_model_single_precision):
    llvm_model, lightgbm_model = llvm_lgbm_model_single_precision
    input_data = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=20 * llvm_model.num_feature(),
            min_size=20 * llvm_model.num_feature(),
        )
    )
    input_data = np.array(input_data, dtype=np.float32).reshape(
        20, llvm_model.num_feature()
    )
    lgbm_result = lightgbm_model.predict(input_data)
    llvm_result = llvm_model.predict(input_data)
    assert llvm_result.dtype == np.float32
    """
    # I set rtol & atol according to:
    https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#numpy.isclose    
    https://github.com/numpy/numpy/issues/10161#issuecomment-852783433
    
    
    """
    np.testing.assert_allclose(
        lgbm_result,
        llvm_result,
        rtol=1e-5,
        atol=1e-8,
    )


@given(data=st.data())
@settings(deadline=None)  # the airline model takes a few seconds to compile
def test_forest_llvm_mode_cat(data, llvm_lgbm_model_cat):
    llvm_model, lgbm_model, cat_bitvec = llvm_lgbm_model_cat

    input_cats = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2**31 - 2),
            min_size=sum(cat_bitvec),
            max_size=sum(cat_bitvec),
        )
    )
    input_floats = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=lgbm_model.num_feature() - sum(cat_bitvec),
            min_size=lgbm_model.num_feature() - sum(cat_bitvec),
        )
    )
    input_data = [
        input_cats.pop() if is_cat else input_floats.pop() for is_cat in cat_bitvec
    ]
    np.testing.assert_array_almost_equal(
        llvm_model.predict([input_data]), lgbm_model.predict([input_data]), decimal=15
    )


def test_multiclass_generated(tmpdir):
    """Check prediction equality on a freshly trained multiclass model"""
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=7,
        n_redundant=7,
        n_classes=10,
        random_state=1337,
    )
    d_train = lightgbm.Dataset(X, label=y)
    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": 10,
    }
    # will result in 3*10 trees
    clf = lightgbm.train(params, d_train, 3)

    model_file = str(tmpdir / "model.txt")
    clf.save_model(model_file)

    lgbm = lightgbm.Booster(model_file=model_file)
    llvm = lleaves.Model(model_file=model_file)
    llvm.compile()

    # check predictions equal on the whole dataset
    np.testing.assert_almost_equal(
        lgbm.predict(X, n_jobs=2), llvm.predict(X, n_jobs=2), decimal=10
    )
    assert lgbm.num_model_per_iteration() == llvm.num_model_per_iteration()


def test_random_forest_classifier(tmpdir):
    centers = [[-4, -4], [4, 4]]
    X, y = make_blobs(n_samples=100, centers=centers, random_state=42)

    # rf = random forest (outputs are averaged over all trees)
    params = {
        "boosting_type": "rf",
        "n_estimators": 7,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
    }
    clf = lightgbm.LGBMClassifier(**params).fit(X, y)
    model_file = str(tmpdir / "model.txt")
    clf.booster_.save_model(model_file)

    lgbm = lightgbm.Booster(model_file=model_file)
    llvm = lleaves.Model(model_file=model_file)
    llvm.compile()

    # check predictions equal on the whole dataset
    np.testing.assert_almost_equal(
        lgbm.predict(X, n_jobs=2), llvm.predict(X, n_jobs=2), decimal=10
    )
    assert lgbm.num_model_per_iteration() == llvm.num_model_per_iteration()


@pytest.mark.parametrize("num_trees", [34, 35])
def test_random_forest_regressor(tmpdir, num_trees):
    n_samples = 1000
    X, y = make_regression(n_samples=n_samples, n_features=5, noise=10.0)

    params = {
        "objective": "regression",
        "n_jobs": 1,
        "boosting_type": "rf",
        "subsample_freq": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "num_leaves": 25,
        "n_estimators": num_trees,
        "min_child_samples": 100,
        "verbose": 0,
    }

    model = lightgbm.LGBMRegressor(**params).fit(X, y)
    model_file = str(tmpdir / "model.txt")
    model.booster_.save_model(model_file)

    lgbm = lightgbm.Booster(model_file=model_file)
    llvm = lleaves.Model(model_file=model_file)
    llvm.compile()

    np.testing.assert_almost_equal(
        lgbm.predict(X, n_jobs=2), llvm.predict(X, n_jobs=2), decimal=10
    )
