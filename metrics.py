# Defininig metrics to compute

import pandas as pd
import numpy as np
from config import *

def compute_metrics(results_df: pd.DataFrame, labels_path: str, fallback: float) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    merged = results_df.merge(labels, on="response_id", how="inner")
 
    if merged.empty:
        print("WARNING: no results matched labels; skipping metrics.")
        return pd.DataFrame()
 
    y = merged["is_correct"].astype(float).to_numpy()
    p = pd.to_numeric(merged["probability"], errors="coerce").fillna(fallback).astype(float).to_numpy()
    eps = 1e-6
    p_clipped = np.clip(p, eps, 1 - eps)
 
    log_loss = float(-(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)).mean())
    brier = float(np.mean((p_clipped - y) ** 2))
 
    # ROC AUC via rank-sum (no sklearn dependency)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos > 0 and n_neg > 0:
        ranks = pd.Series(p_clipped).rank(method="average").to_numpy()
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = float("nan")
 
    # expected calibration error, 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_clipped >= lo) & (p_clipped <= hi if i == 9 else p_clipped < hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p_clipped[mask].mean())
 
    metrics = pd.DataFrame([{
        "prompt_name": PROMPT_NAME,
        "n": len(merged),
        "log_loss": log_loss,
        "brier_score": brier,
        "roc_auc": auc,
        "ece_10_bins": float(ece),
        "mean_probability": float(p_clipped.mean()),
        "label_prevalence": float(y.mean()),
        "parse_success_rate": float(merged["probability"].notna().mean()),
        "mean_elapsed_sec": float(merged["elapsed_sec"].mean()) if "elapsed_sec" in merged else None,
        "mean_attempts": float(merged["attempts"].mean()) if "attempts" in merged else None,
    }])
    return metrics
 