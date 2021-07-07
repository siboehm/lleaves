import concurrent.futures
import math
import os
from ctypes import CFUNCTYPE, POINTER, c_double, c_int
from pathlib import Path

import llvmlite.binding as llvm
import numpy as np

from lleaves import compiler
from lleaves.compiler.ast import scanner
from lleaves.compiler.objective_funcs import get_objective_func
from lleaves.data_processing import (
    data_to_ndarray,
    extract_pandas_traintime_categories,
    ndarray_to_1Darray,
)

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
    _execution_engine = None

    # number of features (=columns)
    _num_feature = None

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

        general_info = scanner.scan_model_file(model_file, general_info_only=True)[
            "general_info"
        ]
        # objective function is implemented as an np.ufunc.
        self.objective_transf = get_objective_func(*general_info["objective"])

    def num_feature(self):
        """
        Returns the number of features used by this model.
        """
        self._assert_is_compiled()
        return self._num_feature

    def compile(self, cache=None):
        """
        Generate the LLVM IR for this model and compile it to ASM.

        This function can be called multiple times, but will only compile once.

        :param cache: Path to a cache file. If this path doesn't exist, binary will be dumped at path after compilation.
                      If path exists, binary will be loaded and compilation skipped.
                      No effort is made to check staleness / consistency.
                      The precise workings of the cache parameter will be subject to future changes.
        """
        if self.is_compiled:
            return

        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        # Create a target machine representing the host
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine()

        if cache is None or not Path(cache).exists():
            # Compile to LLVM IR
            module = self._get_llvm_module()
        else:
            # when loading binary from cache we use a dummy empty module
            module = llvm.parse_assembly("")
        module, self._num_feature = compiler.compile_to_module(self.model_file)

        # Create execution engine for our module
        self._execution_engine = llvm.create_mcjit_compiler(module, target_machine)

        # when caching we dump the executable once the module finished compiling
        def save_to_cache(module, buffer):
            if cache and not Path(cache).exists():
                with open(cache, "wb") as file:
                    file.write(buffer)

        # when caching load the executable if it exists
        def load_from_cache(module):
            if cache and Path(cache).exists():
                return Path(cache).read_bytes()

        self._execution_engine.set_object_cache(
            notify_func=save_to_cache, getbuffer_func=load_from_cache
        )

        # compile IR to ASM
        self._execution_engine.finalize_object()
        self._execution_engine.run_static_constructors()

        # construct entry func
        addr = self._execution_engine.get_function_address("forest_root")
        # Drops GIL during call, re-acquires it after
        self._c_entry_func = ENTRY_FUNC_TYPE(addr)
        self.is_compiled = True

    def predict(self, data, n_jobs=os.cpu_count()):
        """
        Return predictions for the given data.

        The model needs to be compiled before prediction.

        :param data: Pandas df, numpy 2D array or Python list. Shape should be (n_rows, model.num_feature()).
            For fastest speed pass 2D float64 numpy arrays only.
        :param n_jobs: Number of threads to use for prediction. Defaults to number of CPUs. For single-row prediction
            this should be set to 1.
        :return: 1D numpy array, dtype float64
        """
        self._assert_is_compiled()

        # convert all input types to numpy arrays
        data = data_to_ndarray(data, self._pandas_categorical)
        if len(data.shape) != 2 or data.shape[1] != self.num_feature():
            raise ValueError(
                f"Data must be of dimension (N, {self.num_feature()}), is {data.shape}."
            )

        # setup input data
        data, n_predictions = ndarray_to_1Darray(data)
        ptr_data = data.ctypes.data_as(POINTER(c_double))

        # setup output data (predictions)
        predictions = np.empty(n_predictions, dtype=np.float64)
        ptr_preds = predictions.ctypes.data_as(POINTER(c_double))

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
        return self.objective_transf(predictions)

    def _assert_is_compiled(self):
        if not self.is_compiled:
            raise RuntimeError(
                "Functionality only available after compilation. Run model.compile()."
            )
