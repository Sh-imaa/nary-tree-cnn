import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score


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
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    ruc = roc_auc_score(labels, logits[:, 1])
    f1_avg = f1_score(labels, preds, average="macro")
    f1_pos_1 = f1_score(labels, preds, pos_label=1)
    f1_pos_0 = f1_score(labels, preds, pos_label=0)

    metrics = {
        "acc": acc,
        "perc/ppv": perc,
        "npv": perc,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "recall/senstivity": recall,
        "specificity": specificity,
        "f1_avg": f1_avg,
        "f1_pos_1": f1_pos_1,
        "f1_pos_0": f1_pos_0,
        "ruc": ruc,
    }
    return metrics
