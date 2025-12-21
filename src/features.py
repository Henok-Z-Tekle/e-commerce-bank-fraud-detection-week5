import pandas as pd

from .utils import ensure_datetime


def add_time_features(df, signup_col="signup_time", purchase_col="purchase_time"):
    df = df.copy()
    df = ensure_datetime(df, [signup_col, purchase_col])

    df["hour_of_day"] = df[purchase_col].dt.hour
    df["day_of_week"] = df[purchase_col].dt.dayofweek

    time_delta = (df[purchase_col] - df[signup_col]).dt.total_seconds()
    df["time_since_signup"] = time_delta.clip(lower=0)
    df["time_since_signup_hours"] = df["time_since_signup"] / 3600

    return df


def add_user_velocity_features(df, user_col="user_id", time_col="purchase_time", window="24H"):
    df = df.copy()
    df = ensure_datetime(df, [time_col])
    df["_row_id"] = df.index
    df = df.sort_values([user_col, time_col])

    counts = (
        df.groupby(user_col)
        .rolling(window=window, on=time_col)[time_col]
        .count()
        .reset_index(level=0, drop=True)
    )

    window_hours = pd.Timedelta(window).total_seconds() / 3600
    df["txn_count_window"] = counts.fillna(0).astype(int)
    df["txn_velocity_per_hour"] = counts.fillna(0) / window_hours

    df = df.sort_values("_row_id").drop(columns=["_row_id"])
    return df
