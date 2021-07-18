from pathlib import Path

import llvmlite.binding as llvm


def compile_module_to_asm(module, cache_path=None):
    # this initializes the per-process LLVM state.
    # TODO we never call llvm.shutdown(), is this a problem?
    # the execution engine and its modules are collected by gc once they go out of scope
    # some parts of the llvm memory are only deallocated once the process exits
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()

    # Create execution engine for our module
    execution_engine = llvm.create_mcjit_compiler(module, target_machine)

    # when caching we dump the executable once the module finished compiling
    # we only ever have one module, hence we can ignore the 'llvm_module' parameter
    # if the module is already cached we load the bytes without any cache-consistency checks
    if cache_path:
        if Path(cache_path).exists():
            execution_engine.set_object_cache(
                getbuffer_func=lambda _: Path(cache_path).read_bytes()
            )
        else:
            execution_engine.set_object_cache(
                notify_func=lambda _, buffer: Path(cache_path).write_bytes(buffer)
            )

    # compile IR to ASM
    execution_engine.finalize_object()
    execution_engine.run_static_constructors()
    return execution_engine
