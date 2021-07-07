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

    # Ich glaube der cache-Code wird einfacher wenn man explizit zwei Faelle
    # hinschreibt (Cache hit/miss).
    # Wenn ich den Code richtig verstehe, wird gerade bei Cache hit auch
    # "save_to_cache" ausgefuehrt und hat dann Abbruch durch den exists()
    # call? Waere es nicht besser, im Fall von Cache hit einfach set_object_cache ohne
    # notify_func= aufzurufen?
    # when caching we dump the executable once the module finished compiling
    def save_to_cache(module, buffer):
        if cache and not Path(cache).exists():
            with open(cache, "wb") as file:
                file.write(buffer)

    # when caching load the executable if it exists
    def load_from_cache(module):
        if cache and Path(cache).exists():
            return Path(cache).read_bytes()

    execution_engine.set_object_cache(
        notify_func=save_to_cache, getbuffer_func=load_from_cache
    )

    # compile IR to ASM
    execution_engine.finalize_object()
    execution_engine.run_static_constructors()
    return execution_engine
