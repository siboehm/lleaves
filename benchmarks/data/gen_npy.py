import numpy as np
import pandas as pd

from benchmarks.benchmark import NYC_used_columns
from benchmarks.train_NYC_model import feature_enginering

df = pd.read_csv("airline_data_factorized.csv")
airline_X = df.to_numpy(np.float32)
with open("airline.npy", "wb") as f:
    np.save(f, airline_X)

df = pd.read_parquet("yellow_tripdata_2016-01.parquet", columns=NYC_used_columns)
NYC_X = feature_enginering().fit_transform(df).astype(np.float32)
with open("NYC_taxi.npy", "wb") as f:
    np.save(f, NYC_X)

df = pd.read_parquet("mtpl2.parquet")
mtpl2_X = df.to_numpy(np.float32)
with open("mtpl2.npy", "wb") as f:
    np.save(f, mtpl2_X)
