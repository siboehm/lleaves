import os

import llvmlite.binding as llvm
import llvmlite.ir

from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import gen_forest


def compile_to_module(file_path):
    forest = parse_to_ast(file_path)

    ir = llvmlite.ir.Module(name="forest")
    gen_forest(forest, ir)

    module = llvm.parse_assembly(str(ir))
    module.verify()

    if os.environ.get("LLEAVES_PRINT_UNOPTIMIZED_IR") == "1":
        print(module)

    # Create optimizer
    pmb = llvm.PassManagerBuilder()
    pmb.opt_level = 3
    pm_module = llvm.ModulePassManager()
    pm_function = llvm.FunctionPassManager(module)
    # Add optimization passes to module-level optimizer
    pmb.populate(pm_module)
    pmb.populate(pm_function)

    f = module.get_function("forest_root")
    pm_function.initialize()
    pm_function.run(f)
    pm_function.finalize()

    # single pass only, compiler optimizations don't bring much speedup and take time
    pm_module.run(module)

    if os.environ.get("LLEAVES_PRINT_OPTIMIZED_IR") == "1":
        print(module)

    return module
