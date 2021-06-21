import llvmlite.binding as llvm

from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import ir_from_ast


def compile_to_module(file_path):
    forest = parse_to_ast(file_path)
    ir = ir_from_ast(forest)

    module = llvm.parse_assembly(str(ir))
    module.verify()

    # Create optimizer
    pmb = llvm.PassManagerBuilder()
    pmb.opt_level = 3
    pmb.inlining_threshold = 30
    pm_module = llvm.ModulePassManager()
    # Add optimization passes to module-level optimizer
    pmb.populate(pm_module)

    # single pass only, compiler optimizations don't bring much speedup and take time
    pm_module.run(module)
    return module
