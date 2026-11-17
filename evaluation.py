from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_predictions(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "report": classification_report(actual, predicted),
        "confusion_matrix": confusion_matrix(actual, predicted)
    }
