from typing import List

import numpy as np

try:
    from pandas import DataFrame as pd_DataFrame
except ImportError:

    class pd_DataFrame:
        """Dummy class for pandas.DataFrame."""

        pass


def _dataframe_to_ndarray(data, pandas_categories: List[List]):
    """
    Converts the given dataframe into a 2D numpy array, without checking dimensions.
    :param data: 2D pandas dataframe.
    :param pandas_categories: list of lists. For each column a list of all categories in this column.
        The ordering of columns and of categories within each column should match the training dataset.
    :return: 2D np.ndarray, dtype float64 or float32
    """
    cat_cols = list(data.select_dtypes(include=["category"]).columns)
    if len(cat_cols) != len(pandas_categories):
        raise ValueError(
            "The categorical columns in the dataset don't match the categorical columns during training!"
            f"Train had {len(pandas_categories)} categorical columns, data has {len(cat_cols)}"
        )
    for col, category in zip(cat_cols, pandas_categories):
        # we use set_categories to get the same (category -> code) mapping that we used during train
        if list(data[col].cat.categories) != list(category):
            data[col] = data[col].cat.set_categories(category)
    if len(cat_cols):  # cat_cols is list
        data = data.copy()
        # apply (category -> code) mapping. Categories become floats
        data[cat_cols] = (
            data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        )
    data = data.values
    if data.dtype != np.float64 and data.dtype != np.float32:
        data = data.astype(np.float64)
    return data


def _list_to_ndarray(data):
    try:
        data = np.array(data, dtype=np.float64)
    except BaseException:
        raise ValueError("Cannot convert data list to appropriate np array")
    return data


def data_to_ndarray(data, pandas_categorical):
    """

    :param data: Pandas df, numpy 2D array or Python list.
    :param pandas_categorical: list of lists. For each categorical column in dataframe, a list of its categories.
    :return: numpy ndarray
    """
    if isinstance(data, np.ndarray):
        data = data
    elif isinstance(data, pd_DataFrame):
        data = _dataframe_to_ndarray(data, pandas_categorical)
    elif isinstance(data, list):
        data = _list_to_ndarray(data)
    else:
        raise ValueError(
            f"Expecting numpy.ndarray, pandas.DataFrame or Python list, got {type(data)}"
        )
    return data


def ndarray_to_1Darray(data):
    """
    Takes a 2D numpy array, flattens it and converts to float64 if necessary
    :param data: 2D numpy array.
    :return: (1D numpy array (dtype float64), number of rows in original data)
    """
    n_predictions = data.shape[0]
    if data.dtype == np.float64:
        data = np.array(data.reshape(data.size), dtype=np.float64, copy=False)
    else:
        data = np.array(data.reshape(data.size), dtype=np.float64)
    return data, n_predictions
