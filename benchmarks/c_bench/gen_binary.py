import os
import time

import lleaves

model = os.environ["LLEAVES_BENCHMARK_MODEL"]

fcodemodel = os.environ.get("LLEAVES_FCODEMODEL", "large")
finline = os.environ.get("LLEAVES_FINLINE", "True")
assert finline in (None, "True", "False")
fblocksize = os.environ.get("LLEAVES_FBLOCKSIZE", 34)

print(f"Generating {model}.o")

llvm_model = lleaves.Model(
    model_file=f"../../../tests/models/{model}/model.txt",
)
start = time.time()
llvm_model.compile(
    cache=f"../{model}.o",
    fblocksize=int(fblocksize) if fblocksize else None,
    fcodemodel=fcodemodel,
    finline=finline == "True",
)
print(f"Compiling took: {time.time() - start}")
