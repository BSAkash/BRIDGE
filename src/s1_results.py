"""
s1_results.py — Step 1: Stratified results matrix + model rankings.

Input:  data/predictions_zeroshot.jsonl
Output: results/stratified_results.csv
        results/rankings_rbb_all.csv
        results/rankings_rbb_formal.csv
        results/rankings_rbb_informal.csv
        results/ranking_flip.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from config import (PREDICTIONS_PATH, STRATIFIED_RESULTS_PATH,
                    RANKINGS_ALL_PATH, RANKINGS_FORMAL_PATH,
                    RANKINGS_INFORMAL_PATH, RANKING_FLIP_PATH, LABELS)
from utils import load_predictions, macro_f1, recall_per_class, compute_model_rankings


def compute_stratified_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full stratified results matrix:
    model × register × era × source × bias_type → F1 scores
    Also adds register-level summary rows (model × register → F1).
    """
    group_cols = ["model", "register", "era", "source", "bias_type"]
    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["macro_F1"]    = round(macro_f1(g["true_label"], g["predicted_label"]), 4)
        row["n_sentences"] = len(g)
        for c in LABELS:
            c_rows = g[g["true_label"] == c]
            if len(c_rows) > 0:
                row[f"{c}_F1"] = round(
                    (c_rows["predicted_label"] == c).mean(), 4)
            else:
                row[f"{c}_F1"] = None
        rows.append(row)

    # Register-level summaries
    for (model, register), g in df.groupby(["model", "register"]):
        row = {"model": model, "register": register,
               "era": "ALL", "source": "ALL", "bias_type": "ALL"}
        row["macro_F1"]    = round(macro_f1(g["true_label"], g["predicted_label"]), 4)
        row["n_sentences"] = len(g)
        for c in LABELS:
            c_rows = g[g["true_label"] == c]
            row[f"{c}_F1"] = round(
                (c_rows["predicted_label"] == c).mean(), 4) if len(c_rows) > 0 else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols)


def compute_ranking_flip(
    rankings_informal: pd.DataFrame,
    rankings_formal:   pd.DataFrame
) -> pd.DataFrame:
    merged = rankings_informal[["model", "rank", "macro_F1"]].merge(
        rankings_formal[["model", "rank", "macro_F1"]],
        on="model", suffixes=("_informal", "_formal")
    )
    merged["rank_change"] = merged["rank_informal"] - merged["rank_formal"]
    merged["direction"]   = merged["rank_change"].apply(
        lambda x: f"↑ {abs(x)} places" if x > 0
        else (f"↓ {abs(x)} places" if x < 0 else "no change")
    )
    return merged.sort_values("rank_informal")


def main():
    print("Step 1: Stratified Results + Rankings")

    df = load_predictions(PREDICTIONS_PATH)
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models")

    # ── Stratified results ────────────────────────────────────────────────────
    print("\nComputing stratified results...")
    stratified = compute_stratified_results(df)
    stratified.to_csv(STRATIFIED_RESULTS_PATH, index=False)
    print(f"Saved → {STRATIFIED_RESULTS_PATH}  ({len(stratified)} rows)")

    # ── Rankings ──────────────────────────────────────────────────────────────
    print("\nComputing model rankings...")
    for label, register, path in [
        ("All",      None,       RANKINGS_ALL_PATH),
        ("Formal",   "formal",   RANKINGS_FORMAL_PATH),
        ("Informal", "informal", RANKINGS_INFORMAL_PATH),
    ]:
        r = compute_model_rankings(df, register)
        r.to_csv(path, index=False)
        print(f"\n  {label} rankings:")
        print(r[["model", "macro_F1", "rank"]].to_string(index=False))
        print(f"  Saved → {path}")

    # ── Ranking flip ──────────────────────────────────────────────────────────
    r_inf = pd.read_csv(RANKINGS_INFORMAL_PATH)
    r_frm = pd.read_csv(RANKINGS_FORMAL_PATH)
    flip  = compute_ranking_flip(r_inf, r_frm)
    flip.to_csv(RANKING_FLIP_PATH, index=False)
    print(f"\nRanking flip (informal → formal):")
    print(flip[["model", "rank_informal", "rank_formal", "direction"]].to_string(index=False))
    print(f"Saved → {RANKING_FLIP_PATH}")

    # ── Register gap summary ──────────────────────────────────────────────────
    print("\nRegister gap summary:")
    for register in ["formal", "informal"]:
        subset = df[df["register"] == register]
        acc    = (subset["predicted_label"] == subset["true_label"]).mean()
        print(f"  {register:>10}: accuracy = {acc:.4f}")

    print("\n✅ Step 1 complete.")


if __name__ == "__main__":
    main()
