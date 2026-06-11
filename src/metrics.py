"""Evaluation metrics.

- frame-level: precision / recall / F1 / accuracy on the frame grid
- event-level: segment detection F1 with IoU matching (a predicted song counts as a hit
  if IoU with a ground-truth song >= iou_threshold), plus boundary MAE for matched pairs.
Event metrics are computed AFTER post-processing, which is what the user actually sees.
"""
from __future__ import annotations

import numpy as np


def frame_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    p = (pred >= threshold).astype(np.int8)
    t = (target >= 0.5).astype(np.int8)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"frame_precision": prec, "frame_recall": rec, "frame_f1": f1, "frame_acc": acc}


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def event_metrics(pred_segs: list[tuple[float, float]], true_segs: list[tuple[float, float]],
                  iou_threshold: float = 0.5) -> dict:
    """Greedy IoU matching (each gt matched at most once)."""
    matched_true: set[int] = set()
    matched = []
    for ps in sorted(pred_segs):
        best_j, best_iou = -1, 0.0
        for j, ts in enumerate(true_segs):
            if j in matched_true:
                continue
            v = _iou(ps, ts)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_iou >= iou_threshold:
            matched_true.add(best_j)
            matched.append((ps, true_segs[best_j]))
    tp = len(matched)
    fp = len(pred_segs) - tp
    fn = len(true_segs) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    onset_err = [abs(p[0] - t[0]) for p, t in matched]
    offset_err = [abs(p[1] - t[1]) for p, t in matched]
    return {
        "event_precision": prec, "event_recall": rec, "event_f1": f1,
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
        "onset_mae_s": float(np.mean(onset_err)) if onset_err else None,
        "offset_mae_s": float(np.mean(offset_err)) if offset_err else None,
    }
