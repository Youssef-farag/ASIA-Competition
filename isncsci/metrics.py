"""Evaluation metrics for {0, 1, 2} ordinal targets."""

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def compute_ordinal_metrics(y_true_all, y_pred_all):
    """RMSE, MAE, accuracy, within-1 accuracy, and QWK for {0,1,2} ordinal predictions."""
    y_true = np.asarray(y_true_all, dtype=float)
    y_pred = np.asarray(y_pred_all, dtype=float)
    mask = ~np.isnan(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    y_rounded = np.clip(np.round(y_pred), 0, 2).astype(int)
    y_int     = y_true.astype(int)
    acc     = (y_rounded == y_int).mean()
    within1 = (np.abs(y_rounded - y_int) <= 1).mean()
    try:
        qwk = cohen_kappa_score(y_int, y_rounded, weights='quadratic')
    except Exception:
        qwk = float('nan')
    return dict(rmse=rmse, mae=mae, acc=acc, within1_acc=within1, qwk=qwk)


def print_ordinal_metrics(metrics, label=''):
    prefix = f"{label} " if label else ""
    print(f"  {prefix}RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
          f"Acc={metrics['acc']:.3f}  Within-1={metrics['within1_acc']:.3f}  "
          f"QWK={metrics['qwk']:.4f}")
