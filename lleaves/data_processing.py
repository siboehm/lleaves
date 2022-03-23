import json
import os
from ctypes import POINTER, c_double
from typing import List, Optional

import numpy as np

try:
    from pandas import DataFrame as pd_DataFrame
except ImportError:

    class pd_DataFrame:
        """Dummy class for pandas.DataFrame."""

        pass


def _dataframe_to_ndarray(data: pd_DataFrame, pd_traintime_categories: List[List]):
    """
    Converts the given dataframe into a 2D numpy array and converts categorical columns to float.

    Categoricals present in the dataframe are mapped to their float IDs. The `pd_traintime_categories` are used
    to ensure this categorical -> ID mapping is the same as it was in the training dataset.

    :param data: 2D pandas dataframe.
    :param pd_traintime_categories: For each column a list of all categories in this column.
        The ordering of columns and of categories within each column should match the training dataset.

        Example (two columns with two categories each): ``[["a", "b"], ["b", "a"]]``.
        These columns are different and will result in two different mappings ("a" -> 0.0 vs "a" -> 1.0).
    :return: 2D np.ndarray, dtype float64 or float32
    """
    cat_cols = list(data.select_dtypes(include=["category"]).columns)
    if len(cat_cols) != len(pd_traintime_categories):
        raise ValueError(
            "The categorical columns in the dataset don't match the categorical columns during training!"
            f"Train had {len(pd_traintime_categories)} categorical columns, data has {len(cat_cols)}"
        )
    if len(cat_cols):
        data = data.copy()
        for col, category in zip(cat_cols, pd_traintime_categories):
            # we use set_categories to get the same (category -> code) mapping that we used during train
            if list(data[col].cat.categories) != list(category):
                data[col] = data[col].cat.set_categories(category)
        # apply (category -> code) mapping. Categories become floats
        data[cat_cols] = (
            data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        )
    data = data.values
    if data.dtype != np.float64 and data.dtype != np.float32:
        data = data.astype(np.float64)
    return data


def data_to_ndarray(data, pd_traintime_categories: Optional[List[List]] = None):
    """
    Convert the given data to a numpy ndarray

    For pandas dataframes categories are mapped to floats.
    This mapping needs to be the same as it was during model training, which is achieved via ``pandas_categorical``.

    Example for two columns with two categories each: ``pd_traintime_categories = [["a", "b"], ["b", "a"]]``.
    These are two different columns and result in different mappings: "a" -> 0.0, "b" -> 1.0, vs "b" -> 0.0, "a" -> 1.0.

    LightGBM generates this list of lists at traintime like so::

      pd_traintime_categories = [
        list(df[col].cat.categories)
        for col in df.select_dtypes(include=['category']).columns
      ]

    The result is appended via ``json.dump`` to the model.txt under the 'pandas_categorical' key.
    You can extract it from there using :func:`lleaves.data_processing.extract_pandas_traintime_categories`.

    :param data: Pandas dataframe, numpy array or Python list. No dimension checking occurs. If a dataframe is passed
        the number of categorical columns needs to equal ``len(pd_traintime_categories)``.
    :param pd_traintime_categories: For each categorical column in dataframe, a list of its categories.
        The ordering of columns and of categories within each column should match the training dataset.
        Ignored if data is not a pandas DataFrame.

    :return: numpy ndarray
    """
    if isinstance(data, np.ndarray):
        data = data
    elif isinstance(data, pd_DataFrame):
        data = _dataframe_to_ndarray(data, pd_traintime_categories)
    elif isinstance(data, list):
        data = np.array(data, dtype=np.float64)
    else:
        raise ValueError(
            f"Expecting numpy.ndarray, pandas.DataFrame or Python list, got {type(data)}"
        )

    return data


def ndarray_to_ptr(data: np.ndarray):
    """
    Takes a 2D numpy array, converts to float64 if necessary and returns a pointer

    :param data: 2D numpy array. Copying is avoided if possible.
    :return: pointer to 1D array of dtype float64.
    """
    # ravel makes sure we get a contiguous array in memory and not some strided View
    data = data.astype(np.float64, copy=False, casting="same_kind").ravel()
    ptr = data.ctypes.data_as(POINTER(c_double))
    return ptr


def extract_pandas_traintime_categories(file_path):
    """
    Scan the model.txt from the back to extract the 'pandas_categorical' field.

    This is a list of lists that stores the ordering of categories from the pd.DataFrame used for training.
    Storing this list is necessary as LightGBM encodes categories as integer indices and we need to guarantee that
    the mapping (<category string> -> <integer idx>) is the same during inference as it was during training.

    - Example (pandas categoricals were present in training):
      ``pandas_categorical:[["a", "b", "c"], ["b", "c", "d"], ["w", "x", "y", "z"]]``
    - Example (no pandas categoricals during training):
      ``pandas_categorical:[]`` or ``pandas_categorical=null``

    :param file_path: path to model.txt
    :return: list of list. For each pd.categorical column encountered during training, a list of the categories.
    """
    pandas_key = "pandas_categorical:"
    max_offset = os.path.getsize(file_path)
    stepsize = min(1024, max_offset - 1)
    current_offset = stepsize
    lines = []
    # seek backwards from end of file until we have two lines
    # the (pen)ultimate line should be pandas_categorical:XXX
    with open(file_path, "rb") as f:
        while len(lines) < 2 and current_offset < max_offset:
            if current_offset > max_offset:
                current_offset = max_offset
            # read <current_offset>-many Bytes from end of file
            f.seek(-current_offset, os.SEEK_END)
            lines = f.readlines()
            current_offset *= 2

    # pandas_categorical has to be present in the ultimate or penultimate line. Else the model.txt is malformed.
    if len(lines) >= 2:
        last_line = lines[-1].decode().strip()
        if not last_line.startswith(pandas_key):
            last_line = lines[-2].decode().strip()
        if last_line.startswith(pandas_key):
            pandas_categorical = json.loads(last_line[len(pandas_key) :])
            if pandas_categorical is None:
                pandas_categorical = []
            return pandas_categorical
    raise ValueError("Ill formatted model file!")


def extract_model_global_features(file_path):
    """
    Extract number of features, number of classes and number of trees of this model

    :param file_path: path to model.txt
    :return: dict with "n_args", "n_classes", "n_trees"
    """
    res = {}
    with open(file_path, "r") as f:
        for _ in range(3):
            line = f.readline()
            while line and not line.startswith(
                ("max_feature_idx", "num_class", "tree_sizes")
            ):
                line = f.readline()

            if line.startswith("max_feature_idx"):
                res["n_feature"] = int(line.split("=")[1]) + 1
            elif line.startswith("num_class"):
                res["n_class"] = int(line.split("=")[1])
            elif line.startswith("tree_sizes"):
                # `tree_sizes=123 123 123 123`
                res["n_trees"] = len(line.split("=")[1].split(" "))
            else:
                raise ValueError("Ill formatted model file!")
    return res
