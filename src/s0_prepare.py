"""
s0_prepare.py — Step 0: Fix labels, merge per-model files, validate.

Input:  data/merged_predictions.jsonl  (or per-model files in data/)
Output: data/predictions_zeroshot.jsonl (clean, normalized)
"""

import json
import glob
import pandas as pd
from pathlib import Path
from config import (MERGED_PREDICTIONS_PATH, PREDICTIONS_PATH,
                    LABEL_MAP, REGISTER_MAP, ERA_MAP)
from utils import normalize_labels, add_register_era


def load_raw(path: Path) -> pd.DataFrame:
    """Load a JSONL file robustly."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def merge_per_model_files(data_dir: Path) -> pd.DataFrame:
    """Merge individual per-model prediction files if no merged file exists."""
    files = [f for f in glob.glob(str(data_dir / "*.jsonl"))
             if "merged" not in f and "predictions_zeroshot" not in f
             and "external" not in f]
    if not files:
        raise FileNotFoundError(f"No prediction files found in {data_dir}")
    dfs = []
    for f in files:
        tmp = load_raw(Path(f))
        print(f"  Loaded {Path(f).name}: {len(tmp):,} rows")
        dfs.append(tmp)
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Total merged: {len(merged):,} rows")
    return merged


def validate(df: pd.DataFrame) -> None:
    print("\nValidation:")
    print(f"  Total rows:    {len(df):,}")
    print(f"  Models ({df['model'].nunique()}): {sorted(df['model'].unique())}")
    print(f"  Registers:     {df['register'].value_counts().to_dict()}")
    print(f"  Predicted:     {df['predicted_label'].value_counts().to_dict()}")
    print(f"  True:          {df['true_label'].value_counts().to_dict()}")
    print(f"  Bias types:    {df['bias_type'].value_counts().to_dict()}")
    print(f"  Accuracy:      {(df['predicted_label'] == df['true_label']).mean():.4f}")
    print("\n  Per-model accuracy:")
    acc = (df.groupby("model")
             .apply(lambda g: (g["predicted_label"] == g["true_label"]).mean())
             .round(4))
    for model, a in acc.items():
        flag = " ⚠️  near random" if a < 0.40 else ""
        print(f"    {model:<35} {a:.4f}{flag}")


def main():
    from config import DATA_DIR
    print("Step 0: Prepare Predictions")

    # Load raw file
    if MERGED_PREDICTIONS_PATH.exists():
        print(f"Loading {MERGED_PREDICTIONS_PATH}...")
        df = load_raw(MERGED_PREDICTIONS_PATH)
    elif PREDICTIONS_PATH.exists():
        print(f"Loading {PREDICTIONS_PATH} (already processed)...")
        df = load_raw(PREDICTIONS_PATH)
    else:
        print("No merged file found. Merging per-model files...")
        df = merge_per_model_files(DATA_DIR)

    print(f"Loaded: {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Normalize labels
    print("\nNormalizing labels...")
    df = normalize_labels(df)

    # Add register + era if missing
    df = add_register_era(df)

    # Validate
    validate(df)

    # Save
    df.to_json(PREDICTIONS_PATH, orient="records", lines=True)
    print(f"\n✅ Saved → {PREDICTIONS_PATH}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
