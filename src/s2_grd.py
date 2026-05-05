"""
s2_grd.py — Step 2: Complete GRDC metric suite.

Computes:
  - GRD  (Grouped Recall Divergence)
  - DC   (Directional Consistency)
  - RPF  (Register Polarity Flip)
  - CD   (Confusion Drift)
  - CCD  (Capability-Conditional Drift)

Also runs GRD on era and source groupings (generalizability demo).

Input:  data/predictions_zeroshot.jsonl
Output: results/grd_results.json
        results/rrs_per_model_class.csv
        results/confusion_drift.csv
        results/capability_drift.csv
"""

import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, confusion_matrix
from config import (PREDICTIONS_PATH, GRD_RESULTS_PATH, RRS_PATH,
                    CD_PATH, CCD_PATH, LABELS)
from utils import load_predictions, save_json, recall_per_class, macro_f1


# ══════════════════════════════════════════════════════════════════════════════
# CORE: Grouped Recall Divergence
# ══════════════════════════════════════════════════════════════════════════════

def compute_grd(
    df: pd.DataFrame,
    true_col:   str = "true_label",
    pred_col:   str = "predicted_label",
    model_col:  str = "model",
    group_col:  str = "register",
    ref_group:  str = "informal",
    cmp_group:  str = "formal",
) -> dict:
    """
    Compute full GRDC suite for a given grouping variable.

    Returns dict with:
      GRD_global, RPF, RPF_score, per_class metrics,
      per_model RBD, raw RRS dataframe
    """
    classes = sorted(df[true_col].unique())
    models  = sorted(df[model_col].unique())

    # ── Recall per model × group × class ──────────────────────────────────────
    recall_rows = []
    for model in models:
        for group in [ref_group, cmp_group]:
            subset = df[(df[model_col] == model) & (df[group_col] == group)]
            if len(subset) == 0:
                continue
            for c in classes:
                c_rows = subset[subset[true_col] == c]
                if len(c_rows) == 0:
                    continue
                recall = (c_rows[pred_col] == c).sum() / len(c_rows)
                recall_rows.append({
                    "model": model, "group": group,
                    "class": c, "recall": recall, "n": len(c_rows)
                })

    recall_df = pd.DataFrame(recall_rows)

    # ── Register Recall Shift (RRS) per model × class ─────────────────────────
    rrs_rows = []
    for model in models:
        for c in classes:
            ref_val = recall_df[
                (recall_df["model"] == model) &
                (recall_df["group"] == ref_group) &
                (recall_df["class"] == c)
            ]["recall"].values

            cmp_val = recall_df[
                (recall_df["model"] == model) &
                (recall_df["group"] == cmp_group) &
                (recall_df["class"] == c)
            ]["recall"].values

            if len(ref_val) == 0 or len(cmp_val) == 0:
                continue

            rrs = float(ref_val[0]) - float(cmp_val[0])
            rrs_rows.append({
                "model":       model,
                "class":       c,
                "recall_ref":  round(float(ref_val[0]), 4),
                "recall_cmp":  round(float(cmp_val[0]), 4),
                "RRS":         round(rrs, 4),
                "abs_RRS":     round(abs(rrs), 4),
            })

    rrs_df = pd.DataFrame(rrs_rows)

    # ── GRD per class + DC ────────────────────────────────────────────────────
    class_metrics = {}
    for c in classes:
        sub       = rrs_df[rrs_df["class"] == c]
        mean_rrs  = float(sub["RRS"].mean())
        mean_abs  = float(sub["abs_RRS"].mean())
        dc        = abs(mean_rrs) / mean_abs if mean_abs > 0 else 0.0
        class_metrics[c] = {
            "GRD":        round(mean_abs, 4),
            "GRD_signed": round(mean_rrs, 4),
            "DC":         round(dc, 4),
            "direction":  f"{ref_group} > {cmp_group}" if mean_rrs > 0
                          else f"{cmp_group} > {ref_group}",
            "n_models":   len(sub),
            "per_model_RRS": dict(zip(sub["model"], sub["RRS"].round(4))),
        }

    # ── GRD global ────────────────────────────────────────────────────────────
    grd_global = float(np.mean([v["GRD"] for v in class_metrics.values()]))

    # ── RPF ───────────────────────────────────────────────────────────────────
    signs       = {c: np.sign(class_metrics[c]["GRD_signed"]) for c in classes}
    class_pairs = list(combinations(classes, 2))
    flipped     = [(c1, c2) for c1, c2 in class_pairs
                   if signs[c1] != signs[c2] and signs[c1] != 0 and signs[c2] != 0]
    rpf         = 1 if len(flipped) > 0 else 0
    rpf_score   = len(flipped) / len(class_pairs) if class_pairs else 0.0

    # ── RBD per model ─────────────────────────────────────────────────────────
    rbd_model = (rrs_df.groupby("model")["abs_RRS"]
                 .mean().round(4).sort_values(ascending=False).to_dict())

    return {
        "grouping":    {"variable": group_col, "reference": ref_group,
                        "comparison": cmp_group, "n_models": len(models),
                        "n_classes": len(classes)},
        "GRD_global":  round(grd_global, 4),
        "RPF":         int(rpf),
        "RPF_score":   round(rpf_score, 4),
        "flipped_pairs": [(c1, c2) for c1, c2 in flipped],
        "per_class":   class_metrics,
        "per_model_RBD": rbd_model,
        "raw_rrs":     rrs_df,
    }


def print_grd_report(results: dict) -> None:
    g = results["grouping"]
    print(f"\n{'='*55}")
    print(f"GRD Report | Group={g['variable']} | "
          f"Ref={g['reference']} vs Cmp={g['comparison']}")
    print(f"Models={g['n_models']} | Classes={g['n_classes']}")
    print(f"{'='*55}")
    print(f"  GRD_global = {results['GRD_global']:.4f}")
    print(f"  RPF        = {results['RPF']}  "
          f"({'POLARITY FLIP' if results['RPF'] else 'no flip'})")
    print(f"  RPF_score  = {results['RPF_score']:.4f}")
    print(f"\n  {'Class':>12}  {'GRD':>6}  {'DC':>6}  Direction")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*30}")
    for c, m in results["per_class"].items():
        flag = " ← DC=1.0 SYSTEMATIC" if m["DC"] >= 0.999 else ""
        print(f"  {c:>12}  {m['GRD']:>6.4f}  {m['DC']:>6.4f}  "
              f"{m['direction']}{flag}")
    print(f"\n  Per-model RBD:")
    for model, rbd in results["per_model_RBD"].items():
        print(f"    {model:<35} {rbd:.4f}")
    if results["flipped_pairs"]:
        print(f"\n  Flipped class pairs (RPF):")
        for c1, c2 in results["flipped_pairs"]:
            d1 = results["per_class"][c1]["direction"]
            d2 = results["per_class"][c2]["direction"]
            print(f"    {c1} ({d1})  ↔  {c2} ({d2})")


# ══════════════════════════════════════════════════════════════════════════════
# CD: Confusion Drift
# ══════════════════════════════════════════════════════════════════════════════

def compute_cd(
    df: pd.DataFrame,
    ref_group: str = "informal",
    cmp_group: str = "formal",
) -> pd.DataFrame:
    """
    Confusion Drift: how does the confusion matrix change across registers?
    CD(true→pred) = P(pred|true, informal) - P(pred|true, formal)
    Positive CD → more likely to predict this way in informal register.
    """
    labels = sorted(df["true_label"].unique())
    rows   = []

    for model, mg in df.groupby("model"):
        for group in [ref_group, cmp_group]:
            subset = mg[mg["register"] == group]
            if len(subset) == 0:
                continue
            cm      = confusion_matrix(subset["true_label"],
                                       subset["predicted_label"], labels=labels)
            row_sum = cm.sum(axis=1)
            for i, tc in enumerate(labels):
                for j, pc in enumerate(labels):
                    prob = cm[i, j] / row_sum[i] if row_sum[i] > 0 else 0
                    rows.append({
                        "model": model, "register": group,
                        "true_class": tc, "pred_class": pc,
                        "prob": round(prob, 4)
                    })

    cd_long = pd.DataFrame(rows)

    # Pivot to get formal vs informal side by side
    pivot = cd_long.pivot_table(
        index=["model", "true_class", "pred_class"],
        columns="register", values="prob"
    ).reset_index()

    if ref_group in pivot.columns and cmp_group in pivot.columns:
        pivot["CD"] = pivot[ref_group] - pivot[cmp_group]
    else:
        pivot["CD"] = 0.0

    # Aggregate across models
    agg = (pivot.groupby(["true_class", "pred_class"])
           .agg(mean_CD=("CD", "mean"), std_CD=("CD", "std"))
           .reset_index().round(4))

    print("\nConfusion Drift (mean across models):")
    print("Positive = more likely in informal; Negative = more likely in formal")
    sig = agg[agg["mean_CD"].abs() > 0.05].sort_values("mean_CD", ascending=False)
    print(sig.to_string(index=False))

    # Key finding: harmful → harmless drift
    h2h = agg[(agg["true_class"] == "harmful") &
              (agg["pred_class"] == "harmless")]
    if len(h2h) > 0:
        cd_val = h2h["mean_CD"].values[0]
        if cd_val < -0.05:
            print(f"\n  ⚠️  KEY FINDING: harmful→harmless misclassification is "
                  f"{abs(cd_val):.3f} MORE LIKELY in formal text")
            print(f"     Models call harmful formal text 'harmless' at higher rate")

    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# CCD: Capability-Conditional Drift
# ══════════════════════════════════════════════════════════════════════════════

def compute_ccd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capability-Conditional Drift:
    Does stronger model capability predict smaller register sensitivity?
    CCD(c) = Spearman_r(capability, |RRS(c)|)
    CCD ≈ 0  → structural problem, won't self-correct with scale
    CCD < 0  → stronger models less affected (will self-correct)
    CCD > 0  → stronger models MORE affected (scaling makes it worse)
    """
    labels = sorted(df["true_label"].unique())
    rows   = []

    for model, mg in df.groupby("model"):
        cap = macro_f1(mg["true_label"], mg["predicted_label"])
        for c in labels:
            for reg, subset in mg.groupby("register"):
                c_rows = subset[subset["true_label"] == c]
                rec    = (c_rows["predicted_label"] == c).mean() if len(c_rows) > 0 else 0
                rows.append({"model": model, "class": c,
                             "register": reg, "recall": rec, "capability": cap})

    full = pd.DataFrame(rows)

    # Compute RRS per model per class
    rrs_rows = []
    for model in full["model"].unique():
        for c in labels:
            inf = full[(full["model"] == model) & (full["register"] == "informal")
                       & (full["class"] == c)]["recall"].values
            frm = full[(full["model"] == model) & (full["register"] == "formal")
                       & (full["class"] == c)]["recall"].values
            cap = full[full["model"] == model]["capability"].iloc[0]
            if len(inf) > 0 and len(frm) > 0:
                rrs = float(inf[0]) - float(frm[0])
                rrs_rows.append({"model": model, "class": c,
                                 "RRS": rrs, "abs_RRS": abs(rrs),
                                 "capability": cap})

    rrs_df = pd.DataFrame(rrs_rows)

    print("\nCCD (Spearman r between capability and |RRS|):")
    print("  r ≈ 0  → structural, won't self-correct with scale")
    print("  r < 0  → stronger models less affected (self-corrects)")
    print("  r > 0  → stronger models MORE affected (gets worse)\n")

    ccd_results = []
    for c in labels:
        sub = rrs_df[rrs_df["class"] == c]
        if len(sub) < 3:
            continue
        r, p = spearmanr(sub["capability"], sub["abs_RRS"])
        ccd_results.append({"class": c, "spearman_r": round(r, 4),
                             "p_value": round(p, 4),
                             "interpretation": (
                                 "structural — won't self-correct" if abs(r) < 0.3
                                 else "capability helps" if r < 0
                                 else "scaling makes it worse"
                             )})
        flag = "✅ STRUCTURAL" if abs(r) < 0.3 else ""
        print(f"  {c:>10}: r={r:.4f}  p={p:.4f}  {flag}")

    return pd.DataFrame(ccd_results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Step 2: GRDC Metric Suite")

    df = load_predictions(PREDICTIONS_PATH)
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models")

    all_results = {}

    # ── GRD on register (primary) ─────────────────────────────────────────────
    print("\n" + "─"*50)
    print("GRD: G = register (formal vs informal) — PRIMARY")
    results_reg = compute_grd(df, group_col="register",
                              ref_group="informal", cmp_group="formal")
    print_grd_report(results_reg)
    rrs_df = results_reg.pop("raw_rrs")
    rrs_df.to_csv(RRS_PATH, index=False)
    print(f"Saved → {RRS_PATH}")
    all_results["register"] = results_reg

    # ── GRD on era (generalizability demo 1) ─────────────────────────────────
    if "era" in df.columns and df["era"].nunique() > 1:
        print("\n" + "─"*50)
        print("GRD: G = era (generalizability demo)")
        eras = df["era"].value_counts()
        print(f"  Era distribution: {eras.to_dict()}")
        # Use contemporary as ref (most data), pre-1965 as cmp
        results_era = compute_grd(df, group_col="era",
                                  ref_group="contemporary", cmp_group="pre-1965")
        results_era.pop("raw_rrs")
        print_grd_report(results_era)
        all_results["era"] = results_era

    # ── GRD on source (generalizability demo 2) ───────────────────────────────
    if "source" in df.columns and df["source"].nunique() > 1:
        print("\n" + "─"*50)
        print("GRD: G = source (generalizability demo)")
        results_src = compute_grd(df, group_col="source",
                                  ref_group="Social Bias Frames",
                                  cmp_group="On the Books: Jim Crow Laws")
        results_src.pop("raw_rrs")
        print_grd_report(results_src)
        all_results["source"] = results_src

    # ── CD ────────────────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("CD: Confusion Drift")
    cd_df = compute_cd(df)
    cd_df.to_csv(CD_PATH, index=False)
    print(f"Saved → {CD_PATH}")

    # ── CCD ───────────────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("CCD: Capability-Conditional Drift")
    ccd_df = compute_ccd(df)
    ccd_df.to_csv(CCD_PATH, index=False)
    print(f"Saved → {CCD_PATH}")
    all_results["ccd"] = ccd_df.to_dict(orient="records")

    # ── Save all GRD results ──────────────────────────────────────────────────
    save_json(all_results, GRD_RESULTS_PATH)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("HEADLINE RESULTS SUMMARY")
    print("="*55)
    reg = all_results["register"]
    print(f"  GRD_global (register) = {reg['GRD_global']:.4f}")
    print(f"  RPF                   = {reg['RPF']} "
          f"({'POLARITY FLIP DETECTED' if reg['RPF'] else 'no flip'})")
    for c, m in reg["per_class"].items():
        print(f"  DC({c:>10})      = {m['DC']:.4f}  "
              f"({m['direction']}, gap={m['GRD_signed']:+.4f})")

    print("\n✅ Step 2 complete.")


if __name__ == "__main__":
    main()
