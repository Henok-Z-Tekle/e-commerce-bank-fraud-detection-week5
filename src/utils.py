from pathlib import Path

import pandas as pd


def load_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def ensure_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def coerce_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_duplicates(df):
    return df.drop_duplicates().copy()


def summarize_missing(df):
    return df.isna().sum().sort_values(ascending=False)


def class_distribution(series):
    counts = series.value_counts(dropna=False)
    total = counts.sum()
    dist = pd.DataFrame({"count": counts, "percent": (counts / total) * 100})
    return dist
