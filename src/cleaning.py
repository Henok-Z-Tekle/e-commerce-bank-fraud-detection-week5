import pandas as pd

from .utils import coerce_numeric, drop_duplicates, ensure_datetime


def clean_fraud_data(df):
    df = drop_duplicates(df)
    df = ensure_datetime(df, ["signup_time", "purchase_time"])
    df = coerce_numeric(df, ["ip_address", "purchase_value", "age"])

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(exclude=["number"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    df = df.dropna(subset=["signup_time", "purchase_time"])

    if "class" in df.columns:
        df["class"] = df["class"].astype(int)

    return df


def clean_creditcard_data(df):
    df = drop_duplicates(df)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)

    return df
