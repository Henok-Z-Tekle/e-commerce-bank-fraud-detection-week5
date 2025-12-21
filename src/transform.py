import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def split_train_test(df, target_col, test_size=0.2, random_state=42, stratify=True):
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def build_preprocessor(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if not transformers:
        raise ValueError("No features available to preprocess.")

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, num_cols, cat_cols


def apply_preprocessor(preprocessor, X_train, X_test=None):
    X_train_transformed = preprocessor.fit_transform(X_train)
    if X_test is None:
        return X_train_transformed, None
    X_test_transformed = preprocessor.transform(X_test)
    return X_train_transformed, X_test_transformed


def resample_training_data(X_train, y_train, method="smote", random_state=42):
    if method not in {"smote", "undersample"}:
        raise ValueError("method must be 'smote' or 'undersample'")

    if method == "smote":
        if sparse.issparse(X_train):
            raise ValueError("SMOTE does not support sparse inputs; use 'undersample'.")
        sampler = SMOTE(random_state=random_state)
    else:
        sampler = RandomUnderSampler(random_state=random_state)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def class_distribution(y):
    counts = y.value_counts().rename("count")
    percent = (counts / counts.sum()) * 100
    return pd.concat([counts, percent.rename("percent")], axis=1)
