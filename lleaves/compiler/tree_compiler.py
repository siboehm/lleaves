import os

import llvmlite.binding as llvm
import llvmlite.ir

from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import gen_forest
from lleaves.llvm_binding import _get_target_machine


def compile_to_module(
    file_path,
    fblocksize=34,
    finline=True,
    raw_score=False,
    froot_func_name="forest_root",
    use_fp64=True,
):
    forest = parse_to_ast(file_path)
    forest.raw_score = raw_score

    ir = llvmlite.ir.Module(name="forest")
    gen_forest(forest, ir, fblocksize, froot_func_name, use_fp64)

    ir.triple = llvm.get_process_triple()
    module = llvm.parse_assembly(str(ir))
    module.name = str(file_path)
    module.verify()

    if os.environ.get("LLEAVES_PRINT_UNOPTIMIZED_IR") == "1":
        print(module)

    # Create optimizer using new pass manager API (llvmlite >= 0.42)
    target_machine = _get_target_machine()

    # Create pipeline tuning options with opt_level 3 equivalent (speed_level=3)
    pto = llvm.PipelineTuningOptions(speed_level=3, size_level=0)

    if finline:
        # Enable loop unrolling and vectorization for inlining equivalent
        pto.loop_unrolling = True
        pto.loop_vectorization = True

    # Create pass builder and get module pass manager
    pass_builder = llvm.PassBuilder(target_machine, pto)
    pm_module = pass_builder.getModulePassManager()

    # Run optimization passes
    pm_module.run(module, pass_builder)

    if os.environ.get("LLEAVES_PRINT_OPTIMIZED_IR") == "1":
        print(module)

    return module
