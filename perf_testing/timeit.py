import statistics
import time

import lightgbm
import treelite
import treelite_runtime
from sklearn.datasets import load_boston

from lleaves import Model

X, y = load_boston(return_X_y=True)
model_file = "../tests/models/boston_housing/model.txt"

lgbm_model = lightgbm.Booster(model_file=model_file)

llvm_model = Model(model_file=model_file)
llvm_model.compile()

model = treelite.Model.load(model_file, model_format="lightgbm")
model.export_lib(toolchain="gcc", libpath="/tmp/treelite_model.so")
treelite_model = treelite_runtime.Predictor("/tmp/treelite_model.so")


def lgbm_predict(X, i, batchsize):
    return lgbm_model.predict(X[i : i + batchsize])


def llvm_predict(X, i, batchsize):
    return llvm_model.predict(X[i : i + batchsize])


def treelite_predict(X, i, batchsize):
    return treelite_model.predict(treelite_runtime.DMatrix(X[i : i + batchsize]))


for model in [treelite_predict, lgbm_predict, llvm_predict]:
    print(model)
    for batchsize in [1, 10, 30, 50, 100, 300]:
        times = []
        print("Batchsize:", batchsize)
        for _ in range(100):
            start = time.perf_counter_ns()
            for _ in range(30):
                for i in range(50):
                    model(X, i, batchsize)
            times.append((time.perf_counter_ns() - start) / (30 * 50) / 1000)
        print(
            round(statistics.mean(times), 2),
            "μs",
            "±",
            round(statistics.pstdev(times), 2),
            "μs",
        )
