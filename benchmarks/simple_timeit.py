import statistics
import time

import lightgbm
import numpy as np
import onnxmltools
import onnxruntime as rt
import pandas as pd
import treelite
import treelite_runtime
from onnxconverter_common import FloatTensorType
from sklearn.datasets import load_boston
from train_NYC_model import feature_enginering

from lleaves import Model

boston_X, boston_y = load_boston(return_X_y=True)
boston_X = boston_X.astype(np.float32)

used_columns = [
    "fare_amount",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "tpep_pickup_datetime",
    "passenger_count",
]
df = pd.read_parquet("data/yellow_tripdata_2016-01.parquet", columns=used_columns)
NYC_y = df.pop("fare_amount")
NYC_X = feature_enginering().fit_transform(df).astype(np.float32)

model_file_boston = "../tests/models/boston_housing/model.txt"
model_file_NYC = "../tests/models/NYC_taxi/model.txt"


class BenchmarkModel:
    model = None
    name = None

    def __init__(self, lgbm_model_file):
        self.model_file = lgbm_model_file

    def setup(self, data):
        raise NotImplementedError()

    def predict(self, data, index, batchsize):
        self.model.predict(data[index : index + batchsize])

    def __str__(self):
        return self.name


class LGBMModel(BenchmarkModel):
    name = "LightGBM Booster"

    def setup(self, data):
        self.model = lightgbm.Booster(model_file=self.model_file)


class LLVMModel(BenchmarkModel):
    name = "LLeaVes"

    def setup(self, data):
        self.model = Model(model_file=self.model_file)
        self.model.compile()


class TreeliteModel(BenchmarkModel):
    name = "Treelite"

    def setup(self, data):
        treelite_model = treelite.Model.load(self.model_file, model_format="lightgbm")
        treelite_model.export_lib(toolchain="gcc", libpath="/tmp/treelite_model.so")
        self.model = treelite_runtime.Predictor("/tmp/treelite_model.so")

    def predict(self, data, index, batchsize):
        return self.model.predict(treelite_runtime.DMatrix(data[i : i + batchsize]))


class TreeliteModelAnnotatedBranches(TreeliteModel):
    name = "Treelite (Annotated Branches)"

    def setup(self, data):
        treelite_model = treelite.Model.load(self.model_file, model_format="lightgbm")
        annotator = treelite.Annotator()
        annotator.annotate_branch(
            model=treelite_model, dmat=treelite_runtime.DMatrix(data)
        )
        annotator.save(path="/tmp/model-annotation.json")
        treelite_model.export_lib(
            toolchain="gcc",
            libpath="/tmp/treelite_model_with_branches.so",
            params={"annotate_in": "/tmp/model-annotation.json"},
        )
        self.model = treelite_runtime.Predictor("/tmp/treelite_model_with_branches.so")


class ONNXModel(BenchmarkModel):
    name = "ONNX"

    def setup(self, data):
        lgbm_model = lightgbm.Booster(model_file=self.model_file)
        onnx_model = onnxmltools.convert_lightgbm(
            lgbm_model,
            initial_types=[
                (
                    "float_input",
                    FloatTensorType([None, lgbm_model.num_feature()]),
                )
            ],
            target_opset=8,
        )
        onnxmltools.utils.save_model(onnx_model, "/tmp/model.onnx")
        self.model = rt.InferenceSession("/tmp/model.onnx")
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name

    def predict(self, data, index, batchsize):
        return self.model.run(
            [self.label_name], {self.input_name: data[index : index + batchsize]}
        )


if __name__ == "__main__":
    for model_file in [model_file_NYC, model_file_boston]:
        data = NYC_X if model_file == model_file_NYC else boston_X
        print(model_file, "\n")
        for model_class in [
            # TreeliteModel,
            # for some reason, treelight with branch annotation is slower than without
            # TreeliteModelAnnotatedBranches
            # LGBMModel,
            LLVMModel,
            ONNXModel,
        ]:
            model = model_class(model_file)
            print(model)
            model.setup(data)
            for batchsize in [1, 5, 10, 30, 50, 100, 300]:
                times = []
                print("Batchsize:", batchsize)
                for _ in range(100):
                    start = time.perf_counter_ns()
                    for _ in range(30):
                        for i in range(50):
                            model.predict(data, i, batchsize)
                    # calc per-batch times, in μs
                    times.append((time.perf_counter_ns() - start) / (30 * 50) / 1000)
                print(
                    round(statistics.mean(times), 2),
                    "μs",
                    "±",
                    round(statistics.pstdev(times), 2),
                    "μs",
                )
