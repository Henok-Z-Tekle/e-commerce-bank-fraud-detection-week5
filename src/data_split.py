from sklearn.model_selection import train_test_split

def stratified_split(df, target_col, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
