import pandas as pd

from src.features import add_time_features, add_user_velocity_features


def test_add_time_features_creates_columns():
    df = pd.DataFrame({
        "signup_time": ["2020-01-01 00:00:00"],
        "purchase_time": ["2020-01-01 05:00:00"],
    })
    out = add_time_features(df)
    assert "hour_of_day" in out.columns
    assert "day_of_week" in out.columns
    assert "time_since_signup" in out.columns
    assert out.loc[0, "time_since_signup_hours"] == 5


def test_add_user_velocity_features_counts_window():
    df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "purchase_time": [
            "2020-01-01 00:00:00",
            "2020-01-01 12:00:00",
            "2020-01-02 00:00:00",
        ],
    })
    out = add_user_velocity_features(df, window="24H")
    assert "txn_count_window" in out.columns
    assert out.loc[0, "txn_count_window"] == 1
    assert out.loc[1, "txn_count_window"] == 2
