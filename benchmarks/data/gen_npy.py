import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmarks.benchmark import NYC_used_columns  # noqa: E402
from benchmarks.train_NYC_model import feature_enginering  # noqa: E402

df = pd.read_csv("airline_data_factorized.csv")
airline_X = df.to_numpy(np.float64)
np.save("airline.npy", airline_X)

df = pd.read_parquet("yellow_tripdata_2016-01.parquet", columns=NYC_used_columns)
NYC_X = feature_enginering().fit_transform(df).astype(np.float64)
np.save("NYC_taxi.npy", NYC_X)

df = pd.read_parquet("mtpl2.parquet")
mtpl2_X = df.to_numpy(np.float64)
np.save("mtpl2.npy", mtpl2_X)
