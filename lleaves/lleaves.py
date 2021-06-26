import concurrent.futures
import os
from ctypes import CFUNCTYPE, POINTER, c_double, c_int
from pathlib import Path

import llvmlite.binding as llvm
import numpy as np

from lleaves import compiler
from lleaves.compiler.ast import scanner
from lleaves.compiler.objective_funcs import get_objective_func

try:
    from pandas import DataFrame as pd_DataFrame
except ImportError:

    class pd_DataFrame:
        """Dummy class for pandas.DataFrame."""

        pass


class Model:
    """
    The base class of lleaves.
    """

    # machine-targeted compiler & exec engine.
    _execution_engine = None

    # LLVM IR Module
    _IR_module: llvm.ModuleRef = None

    # prediction function, drops GIL on entry
    _c_entry_func = None

    def __init__(self, model_file=None):
        """
        Initialize the uncompiled model.

        :param model_file: Path to the model.txt.
        """
        self.model_file = model_file
        self.is_compiled = False

        scanned_model = scanner.scan_model_file(model_file, general_info_only=True)
        self._general_info = scanned_model["general_info"]
        self._pandas_categorical = scanner.scan_pandas_categorical(model_file)

        # objective function is implemented as an np.ufunc.
        self.objective_transf = get_objective_func(self._general_info["objective"])

    def num_feature(self):
        """
        Returns the number of features used by this model.
        """
        return self._general_info["max_feature_idx"] + 1

    def _get_execution_engine(self):
        """
        Create an empty ExecutionEngine suitable for JIT code generation on
        the host CPU. The engine is reusable for an arbitrary number of
        modules.
        """
        if self._execution_engine:
            return self._execution_engine

        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        # Create a target machine representing the host
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = llvm.parse_assembly("")
        self._execution_engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
        return self._execution_engine

    def _get_llvm_module(self):
        if self._IR_module:
            return self._IR_module
        self._IR_module = compiler.compile_to_module(self.model_file)
        return self._IR_module

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
        # CFUNCTYPE params: void return, pointer to data, pointer to results arr, start_idx, end_idx
        # Drops GIL during call, re-aquires it after
        self._c_entry_func = CFUNCTYPE(
            None, POINTER(c_double), POINTER(c_double), c_int, c_int
        )(addr)
        self.is_compiled = True

    def predict(self, data, n_jobs=os.cpu_count()):
        """
        Return predictions for the given data.

        The model needs to be compiled before prediction.

        :param data: Pandas df, numpy 2D array or Python list. For fastest speed pass 2D float64 numpy arrays only.
        :param n_jobs: Number of threads to use for prediction. Defaults to number of CPUs. For single-row prediction
            this should be set to 1.
        :return: 1D numpy array (dtype float64)
        """
        if not self.is_compiled:
            raise RuntimeError(
                "Model needs to be compiled before prediction. Run model.compile()."
            )

        if isinstance(data, pd_DataFrame):
            data = self._data_from_pandas(data)
        data, n_preds = self._to_1d_ndarray(data)
        ptr_data = data.ctypes.data_as(POINTER(c_double))

        preds = np.zeros(n_preds, dtype=np.float64)
        ptr_preds = preds.ctypes.data_as(POINTER(c_double))
        if n_jobs > 1:
            batchsize = n_preds // n_jobs + (n_preds % n_jobs > 0)

            def f(start_idx):
                self._c_entry_func(
                    ptr_data, ptr_preds, start_idx, min(start_idx + batchsize, n_preds)
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                for i in range(0, n_preds, batchsize):
                    executor.submit(f, i)
        else:
            self._c_entry_func(ptr_data, ptr_preds, 0, n_preds)
        return self.objective_transf(preds)

    def _data_from_pandas(self, data):
        if len(data.shape) != 2 or data.shape[0] < 1:
            raise ValueError("Input data must be 2D and non-empty.")
        cat_cols = list(data.select_dtypes(include=["category"]).columns)
        if len(cat_cols) != len(self._pandas_categorical):
            print(cat_cols, self._pandas_categorical)
            raise ValueError(
                "The categorical features passed don't match the train dataset."
            )
        for col, category in zip(cat_cols, self._pandas_categorical):
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
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float64)
        return data

    def _to_1d_ndarray(self, data):
        if isinstance(data, list):
            try:
                data = np.array(data, dtype=np.float64)
            except BaseException:
                raise ValueError("Cannot convert data list to appropriate np array")

        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expecting list or numpy.ndarray, got {type(data)}")
        if len(data.shape) != 2:
            raise ValueError(
                f"Data must be 2 dimensional, is {len(data.shape)} dimensional"
            )
        n_preds = data.shape[0]
        if data.dtype == np.float64:
            # flatten the array to 1D
            data = np.array(data.reshape(data.size), dtype=np.float64, copy=False)
        else:
            data = np.array(data.reshape(data.size), dtype=np.float64)
        return data, n_preds
