from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.cv_results_
