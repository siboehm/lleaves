from pathlib import Path

import llvmlite.binding as llvm


def compile_module_to_asm(module, cache=None):
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
    save_to_cache, load_from_cache = None, None
    if cache:
        if not Path(cache).exists():

            def save_to_cache(llvm_module, buffer):
                Path(cache).write_bytes(buffer)

        else:

            def load_from_cache(llvm_module):
                return Path(cache).read_bytes()

    execution_engine.set_object_cache(
        notify_func=save_to_cache, getbuffer_func=load_from_cache
    )

    # compile IR to ASM
    execution_engine.finalize_object()
    execution_engine.run_static_constructors()
    return execution_engine
