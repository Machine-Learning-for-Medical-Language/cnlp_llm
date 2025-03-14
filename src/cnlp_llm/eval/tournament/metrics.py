import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def get_best_accuracy(y_true, y_prob):
    # Step 1: Calculate the ROC curve
    _, _, thresholds = roc_curve(y_true, y_prob)

    # Step 2: Calculate accuracy for each threshold
    accuracies = []
    for threshold in thresholds:
        # Predicted labels based on the threshold
        y_pred = (y_prob >= threshold).astype(int)
        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)

    # Step 3: Find the threshold with the maximum accuracy
    best_index = np.argmax(accuracies)
    best_accuracy = accuracies[best_index]

    return best_accuracy


def get_best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(
        y_true=labels,
        y_score=scores,  # type: ignore
    )
    f1s = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ind, threshold in enumerate(thresholds):
            f1 = 2 * precision[ind] * recall[ind] / (precision[ind] + recall[ind])
            f1s.append(f1)

    best_index = np.argmax(np.nan_to_num(f1s))
    best_f1 = f1s[best_index]
    return best_f1


__all__ = ["roc_auc_score", "get_best_accuracy", "get_best_f1"]
