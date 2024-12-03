import concurrent.futures
import math
import os
from ctypes import CFUNCTYPE, POINTER, c_double, c_float, c_int32
from pathlib import Path

import llvmlite.binding
import numpy as np

from lleaves import compiler
from lleaves.data_processing import (
    data_to_ndarray,
    extract_model_global_features,
    extract_pandas_traintime_categories,
    ndarray_to_ptr,
)
from lleaves.llvm_binding import compile_module_to_asm


def get_entry_func_type(use_fp64: bool):
    dtype = c_double if use_fp64 else c_float
    return CFUNCTYPE(
        None,  # return void
        POINTER(dtype),  # pointer to data array
        POINTER(dtype),  # pointer to results array
        c_int32,  # start index
        c_int32,  # end index
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

        :param model_file: Path to the model.txt. Hint: If you have the string representation of the model,
            you can use `tempfile` from the standard library to write the string to a file first.
        """
        self.model_file = model_file
        self.is_compiled = False
        self.use_fp64 = True

        self._pandas_categorical = extract_pandas_traintime_categories(model_file)
        num_attrs = extract_model_global_features(model_file)
        self._n_feature = num_attrs["n_feature"]
        self._n_classes = num_attrs["n_class"]
        self._n_trees = num_attrs["n_trees"]

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

    def num_trees(self):
        """
        Returns the number of trees in this model.
        """
        return self._n_trees

    def compile(
        self,
        cache=None,
        *,
        raw_score=False,
        fblocksize=34,
        fcodemodel="large",
        finline=True,
        froot_func_name="forest_root",
        use_fp64=True,
        target_cpu=None,
        target_cpu_features=None,
    ):
        """
        Generate the LLVM IR for this model and compile it to ASM.

        For most users tweaking the compilation flags (fcodemodel, fblocksize, finline) will be unnecessary
        as the default configuration is already very fast.
        Modifying the flags is useful only if you're trying to squeeze out the last few percent of performance.

        :param cache: Path to a cache file. If this path doesn't exist, binary will be dumped at path after compilation.
            If path exists, binary will be loaded and compilation skipped.
            No effort is made to check staleness / consistency.
        :param raw_score: If true, compile the tree to always return raw predictions, without applying
            the objective function. Equivalent to the `raw_score` parameter of LightGBM's Booster.predict().
        :param fblocksize: Trees are cache-blocked into blocks of this size, reducing the icache miss-rate.
            For deep trees or small caches a lower blocksize is better. For single-row predictions cache-blocking
            adds overhead, set `fblocksize=Model.num_trees()` to disable it.
        :param fcodemodel: The LLVM codemodel. Relates to the maximum offsets that may appear in an ASM instruction.
            One of {"small", "large"}.
            The small codemodel will give speedups for most forests, but will segfault when used for compiling
            very large forests.
        :param finline: Whether or not to inline function. Setting this to False will speed-up compilation time
            significantly but will slow down prediction.
        :param froot_func_name: Name of entry point function in the compiled binary. This is the function to link when
            writing a C function wrapper. Defaults to "forest_root".
        :param use_fp64: If true, compile the model to use fp64 (double) precision, else use fp32 (float).
        :param target_cpu: An optional string specifying the target CPU name to specialize for (defaults to the host's
            cpu name).
        :param target_cpu_features: An optional string specifying the target CPU features to enable (defaults to the
            host's CPU features).
        """
        assert fblocksize > 0
        assert fcodemodel in ("small", "large")
        self.use_fp64 = use_fp64

        if cache is None or not Path(cache).exists():
            module = compiler.compile_to_module(
                self.model_file,
                raw_score=raw_score,
                fblocksize=fblocksize,
                finline=finline,
                froot_func_name=froot_func_name,
                use_fp64=self.use_fp64,
            )
        else:
            # when loading binary from cache we use a dummy empty module
            module = llvmlite.binding.parse_assembly("")

        # keep a reference to the engine to protect it from being garbage-collected
        self._execution_engine = compile_module_to_asm(
            module,
            cache,
            fcodemodel=fcodemodel,
            target_cpu=target_cpu,
            target_cpu_features=target_cpu_features,
        )

        # Drops GIL during call, re-acquires it after
        addr = self._execution_engine.get_function_address(froot_func_name)
        self._c_entry_func = get_entry_func_type(self.use_fp64)(addr)

        self.is_compiled = True

    def predict(self, data, n_jobs=None):
        """
        Return predictions for the given data.

        The model needs to be compiled before prediction.

        :param data: Pandas df, numpy 2D array or Python list. Shape should be (n_rows, model.num_feature()).
            If the datatype is not equal to the model's dtype, the data will be copied. In any case access is read-only.
        :param n_jobs: Number of threads to use for prediction. Defaults to number of CPUs. For single-row prediction
            this should be set to 1.
        :return: 1D numpy array. Datatype is fp64/fp32, depending on the `use_fp64` flag passed to .compile()
        """
        if n_jobs is None:
            n_jobs = os.cpu_count()

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
        # protect against `ctypes.c_int32` silently overflowing and causing SIGSEGV
        if n_predictions >= 2**31 - 1:
            raise ValueError(
                "Prediction is not supported for datasets with >=2^31-1 rows. "
                "Split the dataset into smaller chunks first."
            )

        # setup input data and predictions array
        ptr_data = ndarray_to_ptr(data, use_fp64=self.use_fp64)

        pred_shape = (
            n_predictions if self._n_classes == 1 else (n_predictions, self._n_classes)
        )
        predictions = np.zeros(
            pred_shape, dtype=np.float64 if self.use_fp64 else np.float32
        )
        ptr_preds = ndarray_to_ptr(predictions, use_fp64=self.use_fp64)

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
