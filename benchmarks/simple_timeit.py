import time
from statistics import mean, pstdev

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import onnxruntime as rt
import pandas as pd
import seaborn
import treelite
import treelite_runtime
from onnxconverter_common import FloatTensorType
from sklearn.datasets import load_boston
from train_NYC_model import feature_enginering

from lleaves import Model

boston_X, _ = load_boston(return_X_y=True)
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
NYC_X = feature_enginering().fit_transform(df).astype(np.float32)

df = pd.read_csv("data/airline_data_factorized.csv")
airline_X = df.astype(np.float32)

model_file_boston = "../tests/models/boston_housing/model.txt"
model_file_NYC = "../tests/models/NYC_taxi/model.txt"
model_file_airline = "../tests/models/airline/model.txt"


class BenchmarkModel:
    model = None
    name = None

    def __init__(self, lgbm_model_file):
        self.model_file = lgbm_model_file

    def setup(self, data, n_threads):
        raise NotImplementedError()

    def predict(self, data, index, batchsize, n_threads):
        self.model.predict(data[index : index + batchsize])

    def __str__(self):
        return self.name


class LGBMModel(BenchmarkModel):
    name = "LightGBM Booster"

    def setup(self, data, n_threads):
        self.model = lightgbm.Booster(model_file=self.model_file)

    def predict(self, data, index, batchsize, n_threads):
        self.model.predict(
            data[index : index + batchsize], n_jobs=n_threads if n_threads else None
        )


class LLVMModel(BenchmarkModel):
    name = "LLeaVes"

    def setup(self, data, n_threads):
        self.model = Model(model_file=self.model_file)
        self.model.compile()


class TreeliteModel(BenchmarkModel):
    name = "Treelite"

    def setup(self, data, n_threads):
        treelite_model = treelite.Model.load(self.model_file, model_format="lightgbm")
        treelite_model.export_lib(toolchain="gcc", libpath="/tmp/treelite_model.so")
        self.model = treelite_runtime.Predictor(
            "/tmp/treelite_model.so",
            nthread=n_threads if n_threads != 0 else None,
        )

    def predict(self, data, index, batchsize, n_threads):
        return self.model.predict(treelite_runtime.DMatrix(data[i : i + batchsize]))


class TreeliteModelAnnotatedBranches(TreeliteModel):
    name = "Treelite (Annotated Branches)"

    def setup(self, data, n_threads):
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
        self.model = treelite_runtime.Predictor(
            "/tmp/treelite_model_with_branches.so",
            nthread=n_threads if n_threads != 0 else None,
        )


class ONNXModel(BenchmarkModel):
    name = "ONNX"

    def setup(self, data, n_threads):
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
        options = rt.SessionOptions()
        options.inter_op_num_threads = n_threads
        options.intra_op_num_threads = n_threads
        self.model = rt.InferenceSession("/tmp/model.onnx", sess_options=options)
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name

    def predict(self, data, index, batchsize, n_threads):
        return self.model.run(
            [self.label_name], {self.input_name: data[index : index + batchsize]}
        )


if __name__ == "__main__":
    seaborn.set(rc={"figure.figsize": (11.7, 8.27)})
    batchsizes = [1, 2, 3, 5, 7, 10, 30, 70, 100, 200, 300]
    for model_file, data in zip(
        [model_file_NYC],
        [NYC_X],
    ):
        print(model_file, "\n")
        for n_threads in [0, 1]:
            fig, ax = plt.subplots()
            for model_class in [
                LGBMModel,
                TreeliteModel,
                ONNXModel,
                LLVMModel,
            ]:
                model = model_class(model_file)
                results = {"time (μs)": [], "batchsize": []}
                model.setup(data, n_threads)
                for batchsize in batchsizes:
                    times = []
                    for _ in range(100):
                        start = time.perf_counter_ns()
                        for _ in range(30):
                            for i in range(50):
                                model.predict(data, i, batchsize, n_threads)
                        # calc per-batch times, in μs
                        times.append(
                            (time.perf_counter_ns() - start) / (30 * 50) / 1000
                        )
                    results["time (μs)"] += times
                    results["batchsize"] += len(times) * [batchsize]
                    print(
                        f"{model} (Batchsize {batchsize}): {round(mean(times), 2)}μs ± {round(pstdev(times), 2)}μs"
                    )
                plot = seaborn.lineplot(
                    x="batchsize",
                    y="time (μs)",
                    ci="sd",
                    data=results,
                    label=str(model),
                )
            ax.set(
                xscale="log",
                yscale="log",
                title=f"Per-batch prediction time, n_threads={n_threads}",
                xticks=batchsizes,
                xticklabels=batchsizes,
            )
            plot.figure.savefig(f"{model_file.split('/')[-2]}_{n_threads}.png")
            plt.clf()
