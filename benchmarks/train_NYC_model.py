# code stolen from https://github.com/xhochy/nyc-taxi-fare-prediction-deployment-example
# download the data from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# use the pandas code snippet from here
# https://github.com/xhochy/nyc-taxi-fare-prediction-deployment-example/blob/main/training/Train.ipynb
# to convert the CSV to parquet

import lightgbm
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer


def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.radians(x) for x in (lat1, lng1, lat2, lng2))
    d = (
        np.sin(lat2 / 2 - lat1 / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(lng2 / 2 - lng1 / 2) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(d))  # 6,371 km is the earth radius


def haversine_distance_from_df(df):
    return pd.DataFrame(
        {
            "haversine_distance": haversine_distance(
                df["pickup_latitude"],
                df["pickup_longitude"],
                df["dropoff_latitude"],
                df["dropoff_longitude"],
            )
        }
    )


def split_pickup_datetime(df):
    return pd.DataFrame(
        {
            "pickup_dayofweek": df["tpep_pickup_datetime"].dt.dayofweek,
            "pickup_hour": df["tpep_pickup_datetime"].dt.hour,
            "pickup_minute": df["tpep_pickup_datetime"].dt.minute,
        }
    )


def feature_enginering():
    return make_column_transformer(
        (FunctionTransformer(), ["passenger_count"]),
        (
            FunctionTransformer(func=split_pickup_datetime),
            ["tpep_pickup_datetime"],
        ),
        (
            FunctionTransformer(
                func=haversine_distance_from_df,
            ),
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ],
        ),
    )


if __name__ == "__main__":
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
    y = df.pop("fare_amount")

    # feature_enginering is the example pipeline without the Regressor
    features_as_array = feature_enginering().fit_transform(df)
    train_data = lightgbm.Dataset(features_as_array, label=y)
    model = lightgbm.train({"objective": "regression_l1"}, train_data)
    model.save_model("../tests/models/model.txt")
