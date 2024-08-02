from sklearn.metrics import recall_score, confusion_matrix, make_scorer

def bookmakers_informedness(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Extract true negatives, false positives, false negatives, true positives
    TN, FP, FN, TP = cm.ravel()
    # Compute sensitivity (recall) and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    # Calculate Bookmaker's Informedness
    informedness = sensitivity + specificity - 1
    return informedness


def unweighted_average_recall_score(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Extract true negatives, false positives, false negatives, true positives
    TN, FP, FN, TP = cm.ravel()
    # Compute sensitivity (recall) and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    # Calculate unweighted average recall
    uar = (sensitivity + specificity) * 0.5
    return uar
