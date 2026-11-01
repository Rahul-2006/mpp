import numpy as np

def evaluate(pred, df, grid_m, covered=None):
    """pred: (n,k,2) absolute cells. Distance error uses the top-1 prediction (cell centre to cell centre)."""
    true = np.c_[df.next_x.values, df.next_y.values]
    top1 = pred[:, 0, :]
    hit1 = (top1 == true).all(1)
    hitk = (pred == true[:, None, :]).all(2).any(1)
    err = np.hypot(top1[:, 0] - true[:, 0], top1[:, 1] - true[:, 1]) * grid_m
    cov = np.ones(len(df), bool) if covered is None else covered
    return {
        "exact_acc": hit1.mean(), f"top{pred.shape[1]}_acc": hitk.mean(),
        "MAE_m": err.mean(), "RMSE_m": np.sqrt((err ** 2).mean()), "median_err_m": np.median(err),
        "coverage": cov.mean(), "acc_on_covered": hit1[cov].mean() if cov.any() else np.nan,
    }, err
