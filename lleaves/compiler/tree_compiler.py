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
    pmb.inlining_threshold = 30
    pm_module = llvm.ModulePassManager()
    # Add optimization passes to module-level optimizer
    pmb.populate(pm_module)

    # single pass only, compiler optimizations don't bring much speedup and take time
    pm_module.run(module)

    if os.environ.get("LLEAVES_PRINT_OPTIMIZED_IR") == "1":
        print(module)

    return module
