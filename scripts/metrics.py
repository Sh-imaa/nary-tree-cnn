import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def get_metrics(data, preds, logits=None):
    labels = [t.label for t in data]
    labels = np.array(labels)
    preds = np.array(preds)
    acc = (preds == labels).mean()
    tp = ((preds == labels) & (preds == 1)).sum()
    tn = ((preds == labels) & (preds == 0)).sum()
    fp = ((preds != labels) & (preds == 1)).sum()
    fn = ((preds != labels) & (preds == 0)).sum()

    perc = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_minor_1 = 2 * perc * recall / (perc + recall)
    f1_minor_0 = tn / (tn + 0.5 * (fp + fn))
    roc = roc_auc_score(labels, logits[:, 1])
    fpr, tpr, _ = roc_curve(labels, logits[:, 1])

    metrics = {
        "acc": acc,
        "perc": perc,
        "recall": recall,
        "f1_minor_1": f1_minor_1,
        "f1_minor_0": f1_minor_0,
        "roc": roc,
    }

    plt.plot(fpr, tpr)
    plt.show()
    return metrics
