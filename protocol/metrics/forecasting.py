from __future__ import annotations

import numpy as np


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    safe = denominator.astype(np.float64).copy()
    safe[np.abs(safe) < 1e-12] = 1e-12
    return numerator / safe


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(_safe_divide(diff, true))))
    mspe = float(np.mean(_safe_divide(diff, true) ** 2))

    denom = np.sqrt(np.sum((true - np.mean(true)) ** 2))
    rse = float(np.sqrt(np.sum(diff ** 2)) / denom) if denom > 0 else 0.0

    true_centered = true - np.mean(true, axis=0, keepdims=True)
    pred_centered = pred - np.mean(pred, axis=0, keepdims=True)
    corr_num = np.sum(true_centered * pred_centered, axis=0)
    corr_den = np.sqrt(np.sum(true_centered ** 2, axis=0) * np.sum(pred_centered ** 2, axis=0))
    corr = float(np.mean(_safe_divide(corr_num, corr_den)))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "rse": rse,
        "corr": corr,
    }
