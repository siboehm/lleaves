import concurrent.futures
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
    # machine-targeted compiler & exec engine
    _execution_engine = None

    # LLVM IR Module
    _IR_module: llvm.ModuleRef = None

    # prediction function, drops GIL on entry
    _c_entry_func = None

    def __init__(self, model_file=None):
        self.model_file = model_file

        scanned_model = scanner.scan_model_file(model_file, general_info_only=True)
        self._general_info = scanned_model["general_info"]
        self._pandas_categorical = scanner.scan_pandas_categorical(model_file)

        # objective function is implemented as an np.ufunc.
        self.objective_transf = get_objective_func(self._general_info["objective"])

    def num_feature(self):
        """number of features"""
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

    def save_model_ir(self, filepath):
        """
        Save the optimized LLVM IR to filepath.

        This will be optimized specifically to the target machine.
        You should store this together with the model.txt, as certain model features (like the output function)
        are not stored inside the IR.

        :param filepath: file to save to
        """
        Path(filepath).write_text(str(self._get_llvm_module()))

    def load_model_ir(self, filepath):
        """
        Restore saved LLVM IR.
        Instead of compiling & optimizing the model.txt, the loaded model ir will be used, which saves
        compilation time.

        :param filepath: file to load from
        """
        ir = Path(filepath).read_text()
        module = llvm.parse_assembly(ir)
        self._IR_module = module

    def compile(self):
        """
        Generate the LLVM IR for this model and compile it to ASM
        This function can be called multiple times, but will only compile once.
        """
        if self._c_entry_func:
            return

        # add module and make sure it is ready for execution
        exec_engine = self._get_execution_engine()
        exec_engine.add_module(self._get_llvm_module())
        # run asm codegen
        exec_engine.finalize_object()
        exec_engine.run_static_constructors()

        # construct entry func
        addr = exec_engine.get_function_address("forest_root")
        # CFUNCTYPE params: void return, pointer to data, pointer to results arr, start_idx, end_idx
        # Drops GIL during call, re-aquires it after
        self._c_entry_func = CFUNCTYPE(
            None, POINTER(c_double), POINTER(c_double), c_int, c_int
        )(addr)

    def predict(self, data, n_jobs=4):
        """
        Return predictions for the given data

        For fastest speed, pass the data as a 2D numpy array with dtype float64

        :param data: Pandas df, numpy 2D array or Python list
        :return: 1D numpy array, dtype float64
        """
        self.compile()

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
