import os
import time

import lightgbm
import numpy as np
import onnxmltools
import onnxruntime as rt
import pandas as pd
import treelite
import treelite_runtime
from onnxconverter_common import FloatTensorType

from benchmarks.train_NYC_model import feature_enginering
from lleaves import Model
from lleaves.data_processing import extract_pandas_traintime_categories


def ndarray_from_pandas(data, pandas_categorical=None, target_dtype=np.float64):
    """
    Convert a pandas dataframe to a numpy array

    The main difficulty here is making sure the categoricals are mapped correctly to their float index.
    This is done by passing pandas_categorical, which is saved inside LightGBM's model.txt.
    These were the categories for each categorical in the dataframe during train time.

    :param data: dataframe
    :param pandas_categorical: list of list of categories
    :param target_dtype: np dtype for the resulting array
    """
    cat_cols = list(data.select_dtypes(include=["category"]).columns)
    if pandas_categorical is None:
        pandas_categorical = []
    if len(cat_cols) != len(pandas_categorical):
        raise ValueError(
            "The categorical features passed don't match the train dataset."
        )
    for col, category in zip(cat_cols, pandas_categorical):
        # we use set_categories to get the same (category -> code) mapping that we used during train
        if list(data[col].cat.categories) != list(category):
            data[col] = data[col].cat.set_categories(category)
    if len(cat_cols):  # cat_cols is list
        data = data.copy()
        # apply (category -> code) mapping. Categories become floats
        data[cat_cols] = (
            data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        )
    return data.to_numpy(target_dtype)


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


class TreeliteModel(BenchmarkModel):
    name = "Treelite"

    def _setup(self, data, n_threads):
        # disable thread pinning, which modifies (and never resets!) process-global pthreads state
        os.environ["TREELITE_BIND_THREADS"] = "0"
        self._pandas_categorical = extract_pandas_traintime_categories(self.model_file)
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
        if not isinstance(data, np.ndarray):
            data = ndarray_from_pandas(
                data[index : index + batchsize],
                self._pandas_categorical,
                target_dtype=np.float64,
            )
        else:
            data = data[index : index + batchsize]
        return self.model.predict(treelite_runtime.DMatrix(data))


class ONNXModel(BenchmarkModel):
    name = "ONNX Runtime"

    def _setup(self, data, n_threads):
        self._pandas_categorical = extract_pandas_traintime_categories(self.model_file)
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
        self.label_name = self.model.get_outputs()[-1].name

    def predict(self, data, index, batchsize, n_threads):
        if not isinstance(data, np.ndarray):
            data = ndarray_from_pandas(
                data[index : index + batchsize],
                self._pandas_categorical,
                target_dtype=np.float32,
            )
        else:
            data = data[index : index + batchsize].astype(np.float32)
        return self.model.run([self.label_name], {self.input_name: data})


NYC_used_columns = [
    "fare_amount",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "tpep_pickup_datetime",
    "passenger_count",
]


def run_benchmark(
    model_files, np_data, model_classes, threadcount, batchsizes, n_samples=1000
):
    for model_file, data in zip(model_files, np_data):
        model_name = model_file.split("/")[-2]
        print(model_file, f"\n---- {str.upper(model_name)} --- \n")
        for n_threads in threadcount:
            for model_class in model_classes:
                model = model_class(model_file)
                results = {"time (μs)": [], "batchsize": []}
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
                        f"{model} (Batchsize {batchsize}, nthread {n_threads}): {round(min(times), 2)}μs"
                    )
        print()


if __name__ == "__main__":
    df = pd.read_parquet(
        "data/yellow_tripdata_2016-01.parquet", columns=NYC_used_columns
    )
    NYC_X = feature_enginering().fit_transform(df)

    airline_X = pd.read_csv("data/airline_data_factorized.csv")

    mtpl2_X = pd.read_parquet("data/mtpl2.parquet")
    mtpl2_X[["Region", "VehGas"]] = mtpl2_X[["Region", "VehGas"]].astype("category")

    model_file_NYC = "../tests/models/NYC_taxi/model.txt"
    model_file_airline = "../tests/models/airline/model.txt"
    model_file_mtpl2 = "../tests/models/mtpl2/model.txt"

    run_benchmark(
        model_files=[model_file_NYC, model_file_airline, model_file_mtpl2],
        np_data=[NYC_X, airline_X, mtpl2_X],
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

    run_benchmark(
        model_files=[model_file_mtpl2],
        np_data=[mtpl2_X],
        model_classes=[
            LLVMModel,
            LGBMModel,
            TreeliteModel,
            ONNXModel,
        ],
        threadcount=[4],
        batchsizes=[
            10000,
            100000,
            1000000,
        ],
        n_samples=100,
    )
