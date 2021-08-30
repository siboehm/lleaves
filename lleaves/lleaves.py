import concurrent.futures
import math
import os
from ctypes import CFUNCTYPE, POINTER, c_double, c_int
from pathlib import Path

import llvmlite.binding
import numpy as np

from lleaves import compiler
from lleaves.data_processing import (
    data_to_ndarray,
    extract_n_features_n_classes,
    extract_pandas_traintime_categories,
    ndarray_to_ptr,
)
from lleaves.llvm_binding import compile_module_to_asm

ENTRY_FUNC_TYPE = CFUNCTYPE(
    None,  # return void
    POINTER(c_double),  # pointer to data array
    POINTER(c_double),  # pointer to results array
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
    _n_feature = None
    # number of classes
    _n_classes = None

    # prediction function, drops GIL on entry
    _c_entry_func = None

    def __init__(self, model_file):
        """
        Initialize the uncompiled model.

        :param model_file: Path to the model.txt.
        """
        self.model_file = model_file
        self.is_compiled = False

        self._pandas_categorical = extract_pandas_traintime_categories(model_file)
        num_attrs = extract_n_features_n_classes(model_file)
        self._n_feature = num_attrs["n_feature"]
        self._n_classes = num_attrs["n_class"]

    def num_feature(self):
        """
        Returns the number of features used by this model.
        """
        return self._n_feature

    def num_model_per_iteration(self):
        """
        Returns the number of models per iteration.

        This is equal to the number of classes for multiclass models, else will be 1.
        """
        return self._n_classes

    def compile(self, cache=None):
        """
        Generate the LLVM IR for this model and compile it to ASM.

        This method may not be thread-safe in all cases.

        :param cache: Path to a cache file. If this path doesn't exist, binary will be dumped at path after compilation.
                      If path exists, binary will be loaded and compilation skipped.
                      No effort is made to check staleness / consistency.
                      The precise workings of the cache parameter will be subject to future changes.
        """

        if cache is None or not Path(cache).exists():
            module = compiler.compile_to_module(self.model_file)
        else:
            # when loading binary from cache we use a dummy empty module
            module = llvmlite.binding.parse_assembly("")

        # keep a reference to the engine to protect it from being garbage-collected
        self._execution_engine = compile_module_to_asm(module, cache)

        # Drops GIL during call, re-acquires it after
        addr = self._execution_engine.get_function_address("forest_root")
        self._c_entry_func = ENTRY_FUNC_TYPE(addr)

        self.is_compiled = True

    def predict(self, data, n_jobs=os.cpu_count()):
        """
        Return predictions for the given data.

        The model needs to be compiled before prediction.

        :param data: Pandas df, numpy 2D array or Python list. Shape should be (n_rows, model.num_feature()).
            2D float64 numpy arrays have the lowest overhead.
        :param n_jobs: Number of threads to use for prediction. Defaults to number of CPUs. For single-row prediction
            this should be set to 1.
        :return: 1D numpy array, dtype float64.
            If multiclass model: 2D numpy array of shape (n_rows, model.num_model_per_iteration())
        """
        if not self.is_compiled:
            raise RuntimeError(
                "Functionality only available after compilation. Run model.compile()."
            )

        # convert all input types to numpy arrays
        data = data_to_ndarray(data, self._pandas_categorical)
        n_predictions = data.shape[0]
        if len(data.shape) != 2 or data.shape[1] != self.num_feature():
            raise ValueError(
                f"Data must be of dimension (N, {self.num_feature()}), is {data.shape}."
            )

        # setup input data and predictions array
        ptr_data = ndarray_to_ptr(data)

        pred_shape = (
            n_predictions if self._n_classes == 1 else (n_predictions, self._n_classes)
        )
        predictions = np.zeros(pred_shape, dtype=np.float64)
        ptr_preds = ndarray_to_ptr(predictions)

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
