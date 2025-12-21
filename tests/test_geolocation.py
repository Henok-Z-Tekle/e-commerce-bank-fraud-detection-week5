import pandas as pd

from src.geolocation import analyze_fraud_by_country, merge_ip_country


def test_merge_ip_country_maps_range():
    fraud_df = pd.DataFrame({
        "ip_address": [10, 25, 999],
        "class": [0, 1, 0],
    })
    ip_df = pd.DataFrame({
        "lower_bound_ip_address": [0, 20],
        "upper_bound_ip_address": [19, 30],
        "country": ["A", "B"],
    })

    merged = merge_ip_country(fraud_df, ip_df)
    assert merged.loc[0, "country"] == "A"
    assert merged.loc[1, "country"] == "B"
    assert merged.loc[2, "country"] == "Unknown"


def test_analyze_fraud_by_country_returns_rate():
    df = pd.DataFrame({"country": ["A", "A", "B"], "class": [0, 1, 1]})
    summary = analyze_fraud_by_country(df)
    assert set(summary.columns) == {"country", "count", "fraud_rate"}
