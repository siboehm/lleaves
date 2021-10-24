import numpy as np
import pandas as pd

from benchmarks.benchmark import NYC_used_columns
from benchmarks.train_NYC_model import feature_enginering

df = pd.read_csv("airline_data_factorized.csv")
airline_X = df.to_numpy(np.float64)
np.save("airline.npy", airline_X)

df = pd.read_parquet("yellow_tripdata_2016-01.parquet", columns=NYC_used_columns)
NYC_X = feature_enginering().fit_transform(df).astype(np.float64)
np.save("NYC_taxi.npy", NYC_X)

df = pd.read_parquet("mtpl2.parquet")
mtpl2_X = df.to_numpy(np.float64)
np.save("mtpl2.npy", mtpl2_X)
