import pandas as pd


def ip_to_int(ip_series):
    numeric = pd.to_numeric(ip_series, errors="coerce").fillna(0)
    return numeric.astype("int64")


def merge_ip_country(fraud_df, ip_country_df):
    df = fraud_df.copy()
    ranges = ip_country_df.copy()

    ranges["lower_bound_ip_address"] = pd.to_numeric(
        ranges["lower_bound_ip_address"], errors="coerce"
    )
    ranges["upper_bound_ip_address"] = pd.to_numeric(
        ranges["upper_bound_ip_address"], errors="coerce"
    )

    df["ip_int"] = ip_to_int(df["ip_address"])
    df["_row_id"] = df.index

    ranges = ranges.sort_values("lower_bound_ip_address")
    df = df.sort_values("ip_int")

    merged = pd.merge_asof(
        df,
        ranges,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    valid = merged["ip_int"] <= merged["upper_bound_ip_address"]
    merged.loc[~valid, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")

    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])
    return merged


def analyze_fraud_by_country(df, target_col="class"):
    grouped = (
        df.groupby("country")[target_col]
        .agg(count="count", fraud_rate="mean")
        .sort_values("fraud_rate", ascending=False)
        .reset_index()
    )
    return grouped
