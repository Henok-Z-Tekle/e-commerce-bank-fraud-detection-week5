from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)

    return {
        "F1": f1_score(y_test, y_pred),
        "AUC_PR": auc_pr,
        "Confusion_Matrix": confusion_matrix(y_test, y_pred),
        "Report": classification_report(y_test, y_pred)
    }
