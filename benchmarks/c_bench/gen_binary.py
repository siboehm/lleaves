import os

import lleaves

model = os.environ["LLEAVES_BENCHMARK_MODEL"]
print(f"Generating {model}.o")

llvm_model = lleaves.Model(model_file=f"../../../tests/models/{model}/model.txt")
llvm_model.compile(cache=f"../{model}.o")
