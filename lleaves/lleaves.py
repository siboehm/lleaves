import concurrent.futures
import math
import os
from ctypes import CFUNCTYPE, POINTER, c_double, c_float, c_int
from pathlib import Path

import llvmlite.binding
import numpy as np

from lleaves import compiler
from lleaves.data_processing import (
    data_to_ndarray,
    extract_num_feature,
    extract_pandas_traintime_categories,
    ndarray_to_ptr,
)
from lleaves.llvm_binding import compile_module_to_asm


def get_entry_func_type(dtype):
    dtype = c_double if dtype == "float64" else c_float
    return CFUNCTYPE(
        None,  # return void
        POINTER(dtype),  # pointer to data array
        POINTER(dtype),  # pointer to results array
        c_int,  # start index
        c_int,  # end index
    )


class Model:
    """
    The base class of lleaves.
    """

    # machine-targeted compiler & exec engine.
    # We keep this as an object property to protect the compiled binary from being garbage-collected
    _execution_engine = None

    # number of features (=columns)
    _num_feature = None

    # prediction function, drops GIL on entry
    _c_entry_func = None

    def __init__(self, model_file, dtype="float64"):
        """
        Initialize the uncompiled model.

        :param model_file: Path to the model.txt.
        :param dtype: One of ("float64", "float32"). Determines how many bits are used inside the tree for storing
            thresholds, return values and data. LightGBM uses float64, therefore model predictions can
            differ in float32 mode. Float64 is recommended unless it leads to extra data copies.
        """
        self.model_file = model_file
        self.is_compiled = False
        assert dtype in ("float64", "float32")
        self.dtype = dtype

        self._pandas_categorical = extract_pandas_traintime_categories(model_file)
        self._num_feature = extract_num_feature(model_file)

    def num_feature(self):
        """
        Returns the number of features used by this model.
        """
        return self._num_feature

    def compile(self, cache=None):
        """
        Generate the LLVM IR for this model and compile it to ASM.

        This method is not thread-safe and shouldn't be called concurrently.

        :param cache: Path to a cache file. If this path doesn't exist, binary will be dumped at path after compilation.
                      If path exists, binary will be loaded and compilation skipped.
                      No effort is made to check staleness / consistency.
                      The precise workings of the cache parameter will be subject to future changes.
        """

        if cache is None or not Path(cache).exists():
            module = compiler.compile_to_module(
                self.model_file, double_precision=self.dtype == "float64"
            )
        else:
            # when loading binary from cache we use a dummy empty module
            module = llvmlite.binding.parse_assembly("")

        # keep a reference to the engine to protect it from being garbage-collected
        self._execution_engine = compile_module_to_asm(module, cache)

        # Drops GIL during call, re-acquires it after
        addr = self._execution_engine.get_function_address("forest_root")
        self._c_entry_func = get_entry_func_type(self.dtype)(addr)

        self.is_compiled = True

    def predict(self, data, n_jobs=os.cpu_count()):
        """
        Return predictions for the given data.

        The model needs to be compiled before prediction.

        :param data: Pandas df, numpy 2D array or Python list. Shape should be (n_rows, model.num_feature()).
            If the datatype is not equal to the model's dtype, the data will be copied. In any case access is read-only.
        :param n_jobs: Number of threads to use for prediction. Defaults to number of CPUs. For single-row prediction
            this should be set to 1.
        :return: 1D numpy array, dtype float64 / float32.
        """
        if not self.is_compiled:
            raise RuntimeError(
                "Functionality only available after compilation. Run model.compile()."
            )

        # convert all input types to numpy arrays
        data = data_to_ndarray(data, self._pandas_categorical, dtype=self.dtype)
        n_predictions = data.shape[0]
        if len(data.shape) != 2 or data.shape[1] != self.num_feature():
            raise ValueError(
                f"Data must be of dimension (N, {self.num_feature()}), is {data.shape}."
            )

        # setup input data and predictions array
        ptr_data = ndarray_to_ptr(data, dtype=self.dtype)
        predictions = np.zeros(
            n_predictions, dtype=np.float64 if self.dtype == "float64" else np.float32
        )
        ptr_preds = ndarray_to_ptr(predictions, dtype=self.dtype)

        if n_jobs == 1:
            self._c_entry_func(ptr_data, ptr_preds, 0, n_predictions)
        else:
            batchsize = math.ceil(n_predictions / n_jobs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                for i in range(0, n_predictions, batchsize):
                    executor.submit(
                        lambda start_idx: self._c_entry_func(
                            ptr_data,
                            ptr_preds,
                            start_idx,
                            min(start_idx + batchsize, n_predictions),
                        ),
                        i,
                    )
        return predictions
