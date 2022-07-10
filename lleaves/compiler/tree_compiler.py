import os

import llvmlite.binding as llvm
import llvmlite.ir

from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import gen_forest


def compile_to_module(
    file_path,
    fblocksize=34,
    finline=True,
    raw_score=False,
    froot_func_name="forest_root",
):
    forest = parse_to_ast(file_path)
    forest.raw_score = raw_score

    ir = llvmlite.ir.Module(name="forest")
    gen_forest(forest, ir, fblocksize, froot_func_name)

    ir.triple = llvm.get_process_triple()
    module = llvm.parse_assembly(str(ir))
    module.name = str(file_path)
    module.verify()

    if os.environ.get("LLEAVES_PRINT_UNOPTIMIZED_IR") == "1":
        print(module)

    # Create optimizer
    pmb = llvm.PassManagerBuilder()
    pmb.opt_level = 3

    if finline:
        # if inline_threshold is set LLVM inlines, precise value doesn't seem to matter
        pmb.inlining_threshold = 1

    pm_module = llvm.ModulePassManager()
    # Add optimization passes to module-level optimizer
    pmb.populate(pm_module)

    # single pass only, compiler optimizations don't bring much speedup and take time
    pm_module.run(module)

    if os.environ.get("LLEAVES_PRINT_OPTIMIZED_IR") == "1":
        print(module)

    return module
