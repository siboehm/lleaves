import json
from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm

from lleaves.tree_compiler import ir_from_model_file
from lleaves.tree_compiler import parser


class Model:
    # machine-targeted compiler & exec engine
    _execution_engine = None
    # IR representation of model
    _ir_module = None
    # compiled representation of IR
    _compiled_module = None
    compiled = False
    _c_entry_func = None

    def __init__(self, *, model_file=None):
        self.model_file = model_file
        self._general_info = parser.parse_model_file(model_file)["general_info"]

    @property
    def n_features(self):
        """number of features"""
        return self._general_info["max_feature_idx"] + 1

    @property
    def ir_module(self):
        if not self._ir_module:
            self._ir_module = ir_from_model_file(self.model_file)
        return self._ir_module

    @property
    def execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU. The engine is reusable for an arbitrary number of
        modules.
        """
        if not self._execution_engine:
            llvm.initialize()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()

            # Create a target machine representing the host
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            # And an execution engine with an empty backing module
            backing_mod = llvm.parse_assembly("")
            self._execution_engine = llvm.create_mcjit_compiler(
                backing_mod, target_machine
            )
        return self._execution_engine

    def compile(self):
        """
        Generate the LLVM IR for this model and compile it to ASM
        This function can be called multiple time, but will only compile once.
        """
        if not self._compiled_module:
            # Create a LLVM module object from the IR
            module = llvm.parse_assembly(str(self.ir_module))
            module.verify()

            # add module and make sure it is ready for execution
            self.execution_engine.add_module(module)
            self.execution_engine.finalize_object()
            self.execution_engine.run_static_constructors()
            self._compiled_module = module

            # construct entry func
            addr = self._execution_engine.get_function_address("forest_root")
            self._c_entry_func = CFUNCTYPE(c_double, *(self.n_features * (c_double,)))(
                addr
            )

    def predict(self, arrs: list[list[float]]):
        self.compile()
        return [self._c_entry_func(*arr) for arr in arrs]
