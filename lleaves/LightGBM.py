import json
from ctypes import CFUNCTYPE, c_double
from lleaves.tree_compiler import ir_from_json
import llvmlite.binding as llvm


class LGBM:
    # IR of tree
    _llvm_ir = None
    _llvm_is_initialized = False
    _execution_engine = None
    _compiled_module = None
    _c_entry_func = None

    def __init__(self, *, file_path=None, json_str=None):
        if file_path:
            with open(file_path) as f:
                self.model_json = json.load(f)
        else:
            self.model_json = json.loads(json_str)

        self.compiled = False

    @property
    def n_args(self):
        return len(self.model_json["feature_names"])

    def _init_codegen(self):
        if not self._llvm_is_initialized:
            llvm.initialize()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()

    @property
    def llvm_ir(self):
        if not self._llvm_ir:
            self._llvm_ir = ir_from_json(self.model_json)
        return self._llvm_ir

    @property
    def execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        if not self._execution_engine:
            self._init_codegen()

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
        # Create a LLVM module object from the IR
        module = llvm.parse_assembly(self.llvm_ir)
        module.verify()

        # add module and make sure it is ready for execution
        self._execution_engine.add_module(module)
        self._execution_engine.finalize_object()
        self._execution_engine.run_static_constructors()
        self._compiled_module = module

        # construct entry function
        addr = self._execution_engine.get_function_address("root_node")
        self._c_entry_func = CFUNCTYPE(c_double, *(self.n_args * (c_double,)))(addr)

    def __call__(self, arr: list[float]):
        assert len(arr) == self.n_args
        return self._c_entry_func(arr)
