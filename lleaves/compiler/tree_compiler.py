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

    # Create optimizer using New Pass Manager API (llvmlite >= 0.45)
    # Initialization is automatic in llvmlite >= 0.45, but target/asmprinter init still needed
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    target = llvm.Target.from_triple(llvm.get_process_triple())
    try:
        features = llvm.get_host_cpu_features().flatten()
    except RuntimeError:
        features = ""
    target_machine = target.create_target_machine(
        cpu=llvm.get_host_cpu_name(),
        features=features,
        reloc="pic",
        codemodel="large",
    )

    pto = llvm.PipelineTuningOptions(speed_level=3, size_level=0)
    if finline:
        pto.inlining_threshold = 1

    pb = llvm.PassBuilder(target_machine, pto)
    pm_module = pb.getModulePassManager()
    pm_module.run(module, pb)

    if os.environ.get("LLEAVES_PRINT_OPTIMIZED_IR") == "1":
        print(module)

    return module
