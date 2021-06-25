import pickle
import time
from pathlib import Path
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

from benchmarks.train_NYC_model import feature_enginering
from lleaves import Model


class BenchmarkModel:
    model = None
    name = None

    def __init__(self, lgbm_model_file):
        self.model_file = lgbm_model_file

    def setup(self, data, n_threads):
        start = time.perf_counter()
        self._setup(data, n_threads)
        print(f"{self.name} setup: {round(time.perf_counter() - start, 2)}")

    def _setup(self, data, n_threads):
        raise NotImplementedError()

    def predict(self, data, index, batchsize, n_threads):
        return self.model.predict(data[index : index + batchsize], n_jobs=n_threads)

    def __str__(self):
        return self.name


class LGBMModel(BenchmarkModel):
    name = "LightGBM Booster"

    def _setup(self, data, n_threads):
        self.model = lightgbm.Booster(model_file=self.model_file)


class LLVMModel(BenchmarkModel):
    name = "lleaves"

    def _setup(self, data, n_threads):
        self.model = Model(model_file=self.model_file)
        self.model.compile()


class LLVMModelSingle(BenchmarkModel):
    name = "lleaves (single-threaded)"

    def _setup(self, data, n_threads):
        self.model = Model(model_file=self.model_file)
        self.model.compile()

    def predict(self, data, index, batchsize, n_threads):
        return self.model.predict(data[index : index + batchsize], n_jobs=1)


class TreeliteModel(BenchmarkModel):
    name = "Treelite"

    def _setup(self, data, n_threads):
        treelite_model = treelite.Model.load(self.model_file, model_format="lightgbm")
        treelite_model.export_lib(
            toolchain="gcc",
            libpath="/tmp/treelite_model.so",
            params={"parallel_comp": 4},
            verbose=False,
        )
        self.model = treelite_runtime.Predictor(
            "/tmp/treelite_model.so",
            nthread=n_threads,
        )

    def predict(self, data, index, batchsize, n_threads):
        return self.model.predict(
            treelite_runtime.DMatrix(data[index : index + batchsize])
        )


class ONNXModel(BenchmarkModel):
    name = "ONNX Runtime"

    def _setup(self, data, n_threads):
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


def get_color(libname):
    libkey = libname.lower()
    d = {
        "lleaves": "red",
        "lightgbm": "sandybrown",
        "treelite": "mediumseagreen",
        "onnx": "cornflowerblue",
    }
    for key, value in d.items():
        if key in libkey:
            return value


def save_plots(results_full, n_threads, model_files, batchsizes):
    for n_thread in n_threads:
        fig, axs = plt.subplots(ncols=2, nrows=1)
        fig.set_size_inches(16, 9)
        keys = sorted(results_full.keys())
        for count, model_file in enumerate(model_files):
            model_name = model_file.split("/")[-2]
            for key in keys:
                if key.startswith(f"{model_name}_{n_thread}"):
                    if "(single-threaded)" in key:
                        continue
                    seaborn.lineplot(
                        x="batchsize",
                        y="time (μs)",
                        ci="sd",
                        data=results_full[key],
                        ax=axs[count],
                        color=get_color(key),
                        label=key.split("_")[3] if count == 0 else None,
                    )
            axs[count].set(
                xscale="log",
                yscale="log",
                title=model_name,
                xticks=batchsizes,
                xticklabels=batchsizes,
                xlim=(1, None) if n_thread == 1 else None,
            )
        plt.savefig(f"{n_thread}.png", bbox_inches="tight")


NYC_used_columns = [
    "fare_amount",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "tpep_pickup_datetime",
    "passenger_count",
]


def load_results():
    datafile = Path("data/results.pkl")
    res = {}
    if datafile.exists():
        with open(datafile, "rb") as file:
            res = pickle.load(file)
    return res


def save_results(results):
    datafile = Path("data/results.pkl")
    with open(datafile, "wb") as file:
        pickle.dump(results, file)


def run_benchmark(
    model_files, np_data, model_classes, threadcount, batchsizes, n_samples=1000
):
    results_full = load_results()
    for model_file, data in zip(model_files, np_data):
        model_name = model_file.split("/")[-2]
        print(model_file, f"\n---- {str.upper(model_name)} --- \n")
        for n_threads in threadcount:
            for model_class in model_classes:
                model = model_class(model_file)
                results = {"time (μs)": [], "batchsize": []}
                results_full[f"{model_name}_{n_threads}_{model}"] = results
                model.setup(data, n_threads)
                for batchsize in batchsizes:
                    times = []
                    for _ in range(n_samples):
                        start = time.perf_counter_ns()
                        model.predict(data, 0, batchsize, n_threads)
                        # calc per-batch times, in μs
                        times.append((time.perf_counter_ns() - start) / 1000)
                    results["time (μs)"] += times
                    results["batchsize"] += len(times) * [batchsize]
                    print(
                        f"{model} (Batchsize {batchsize}, nthread {n_threads}): {round(mean(times), 2)}μs ± {round(pstdev(times), 2)}μs"
                    )
    save_results(results_full)
    save_plots(results_full, threadcount, model_files, batchsizes)


if __name__ == "__main__":
    df = pd.read_parquet(
        "data/yellow_tripdata_2016-01.parquet", columns=NYC_used_columns
    )
    NYC_X = feature_enginering().fit_transform(df).astype(np.float32)

    df = pd.read_csv("data/airline_data_factorized.csv")
    airline_X = df.to_numpy(np.float32)

    model_file_NYC = "../tests/models/NYC_taxi/model.txt"
    model_file_airline = "../tests/models/airline/model.txt"

    run_benchmark(
        model_files=[model_file_NYC, model_file_airline],
        np_data=[None, None],
        model_classes=[
            LLVMModel,
            LLVMModelSingle,
            LGBMModel,
            TreeliteModel,
            ONNXModel,
        ],
        threadcount=[4],
        batchsizes=[
            1000,
            3000,
            5000,
            10000,
            100000,
            300000,
            1000000,
        ],
        n_samples=100,
    )

    run_benchmark(
        model_files=[model_file_NYC, model_file_airline],
        np_data=[NYC_X, airline_X],
        model_classes=[
            LLVMModel,
            LGBMModel,
            TreeliteModel,
            ONNXModel,
        ],
        threadcount=[1],
        batchsizes=[1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300],
        n_samples=20000,
    )
