"""Residue-pooled metrics, with AP and trapezoidal PR area explicitly separated."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, matthews_corrcoef, f1_score

def metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction arrays")
    keep = y != -1
    y, p = y[keep], p[keep]
    out = {"residues": int(len(y)), "binding": int((y==1).sum()), "non_binding": int((y==0).sum())}
    if not len(y):
        return dict(out, AUC=None, AUPRC=None, PR_AUC_trapezoid=None, MCC=None, F1=None, ACC=None)
    pred = (p > threshold).astype(int)  # matches the original training/evaluation code
    both = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, p) if (y==1).any() else (None,None,None)
    out.update(
        AUC=float(roc_auc_score(y,p)) if both else None,
        AUPRC=float(average_precision_score(y,p)) if (y==1).any() else None,
        PR_AUC_trapezoid=float(auc(recall,precision)) if precision is not None else None,
        MCC=float(matthews_corrcoef(y,pred)), F1=float(f1_score(y,pred,zero_division=0)),
        ACC=float((y==pred).mean()))
    return out
