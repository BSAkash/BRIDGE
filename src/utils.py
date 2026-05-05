"""
utils.py — Shared utilities used across all pipeline steps.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from config import LABELS, LABEL_MAP, REGISTER_MAP, ERA_MAP


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions JSONL. Handles both pandas and manual parsing."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["correct"] = (df["predicted_label"] == df["true_label"]).astype(int)
    return df


def save_json(obj: dict, path) -> None:
    def convert(o):
        if isinstance(o, (np.integer,)):    return int(o)
        if isinstance(o, (np.floating,)):   return float(o)
        if isinstance(o, (np.ndarray,)):    return o.tolist()
        if isinstance(o, (np.bool_,)):      return bool(o)
        raise TypeError(f"Not serializable: {type(o)}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)
    print(f"Saved → {path}")


# ── Label normalization ───────────────────────────────────────────────────────

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize predicted_label, true_label, bias_type to standard values."""
    for col in ["predicted_label", "true_label", "bias_type"]:
        if col not in df.columns:
            continue
        df[col] = (df[col].astype(str).str.strip()
                   .map(lambda x: LABEL_MAP.get(x, "unknown")))
    # Remove unknowns
    before = len(df)
    df = df[df["predicted_label"] != "unknown"].copy()
    df = df[df["true_label"]      != "unknown"].copy()
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed:,} unknown rows")
    df["correct"] = (df["predicted_label"] == df["true_label"]).astype(int)
    return df


def add_register_era(df: pd.DataFrame) -> pd.DataFrame:
    """Add register and era columns from source column."""
    if "source" in df.columns:
        if "register" not in df.columns:
            df["register"] = df["source"].map(REGISTER_MAP).fillna("formal")
        if "era" not in df.columns:
            df["era"] = df["source"].map(ERA_MAP).fillna("modern")
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def macro_f1(true, pred, labels=None):
    labels = labels or LABELS
    return f1_score(true, pred, labels=labels, average="macro", zero_division=0)


def recall_per_class(true, pred, labels=None):
    labels = labels or LABELS
    cm      = confusion_matrix(true, pred, labels=labels)
    row_sum = cm.sum(axis=1)
    recalls = np.where(row_sum > 0, cm.diagonal() / row_sum, 0.0)
    return dict(zip(labels, recalls))


def compute_model_rankings(df: pd.DataFrame, register: str = None) -> pd.DataFrame:
    """Compute model rankings by macro-F1, optionally filtered by register."""
    subset = df[df["register"] == register] if register else df
    rows   = []
    for model, g in subset.groupby("model"):
        f1 = macro_f1(g["true_label"], g["predicted_label"])
        rows.append({"model": model, "macro_F1": round(f1, 4)})
    out = pd.DataFrame(rows)
    out["rank"] = out["macro_F1"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank").reset_index(drop=True)
