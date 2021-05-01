import ctypes
from ctypes import CFUNCTYPE, c_float, c_int
from pathlib import Path

import llvmlite.binding as llvm

# All these initializations are required for code generation!
import numpy as np

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

llvm_ir = Path("model.ll").read_text("utf-8")
print(llvm_ir)


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


engine = create_execution_engine()
mod = compile_ir(engine, llvm_ir)

# Look up the function pointer (a Python int)
func_ptr = engine.get_function_address("forest_root")

# Run the function via ctypes
cfunc = CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
)(func_ptr)
args = np.array([[0.0, 9.0, 0.0]], dtype=np.float64)
args_ptr = args.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
preds = np.zeros(1, dtype=np.float64)
ptr_preds = preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
cfunc(args_ptr, ptr_preds)
print(f"forest_root({', '.join(map(str, args))}) = {preds}")
