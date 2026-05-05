"""
s3_eil_irt.py — Step 3: EIL + 2PL IRT (Supplementary / Appendix material).

Note: EIL requires 20+ models for reliable MI estimation.
With 8 models, EIL = 0 is expected. We include this as a supplementary
analysis and acknowledge the limitation in the paper.

The register_gap and RBD (from s2_grd.py) are the primary metrics.
EIL is reported in the appendix with its limitation noted.

Input:  data/predictions_zeroshot.jsonl
Output: results/eil_results.json
        results/irt_results.csv
        results/construct_shift.json
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from config import (PREDICTIONS_PATH, EIL_RESULTS_PATH,
                    IRT_RESULTS_PATH, CONSTRUCT_SHIFT_PATH, LABELS)
from utils import load_predictions, save_json


# ══════════════════════════════════════════════════════════════════════════════
# EIL
# ══════════════════════════════════════════════════════════════════════════════

def mi_binary_continuous(binary, continuous, n_bins=3):
    try:
        binned = pd.qcut(continuous, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        binned = pd.cut(continuous, bins=n_bins, labels=False)
    mask = ~np.isnan(binned.astype(float))
    b, c = binary[mask].astype(int), binned[mask].astype(int)
    if len(b) < 20 or len(np.unique(c)) < 2:
        return 0.0
    joint = pd.crosstab(b, c, normalize=True).values
    px    = joint.sum(axis=1, keepdims=True)
    py    = joint.sum(axis=0, keepdims=True)
    pxpy  = px * py
    valid = (joint > 0) & (pxpy > 0)
    return max(0.0, float(np.sum(joint[valid] * np.log(joint[valid] / pxpy[valid]))))


def get_capability(df, balanced=True):
    rows = []
    for model, mg in df.groupby("model"):
        if balanced:
            f_f1 = f1_score(
                mg[mg["register"]=="formal"]["true_label"],
                mg[mg["register"]=="formal"]["predicted_label"],
                labels=LABELS, average="macro", zero_division=0)
            i_f1 = f1_score(
                mg[mg["register"]=="informal"]["true_label"],
                mg[mg["register"]=="informal"]["predicted_label"],
                labels=LABELS, average="macro", zero_division=0)
            cap = (f_f1 + i_f1) / 2
        else:
            cap = f1_score(mg["true_label"], mg["predicted_label"],
                           labels=LABELS, average="macro", zero_division=0)
        rows.append({"model": model, "capability": round(cap, 4)})
    cap_df = pd.DataFrame(rows).sort_values("capability", ascending=False)
    cap_df["capability_bin"] = pd.qcut(
        cap_df["capability"], q=3, labels=["low","mid","high"], duplicates="drop")
    return cap_df


def compute_eil(df, cap_df, n_perm=500, n_boot=500):
    cap_lookup = cap_df.set_index("model")["capability"].to_dict()
    df = df.copy()
    df["model_capability"] = df["model"].map(cap_lookup)
    df = df.dropna(subset=["model_capability"])

    correct   = df["correct"].values.astype(int)
    reg_enc   = (df["register"] == "formal").astype(float).values
    capability= df["model_capability"].values

    try:
        cap_bins = pd.qcut(capability, q=3, labels=False, duplicates="drop")
    except Exception:
        cap_bins = pd.cut(capability, bins=3, labels=False)
    cap_bins = cap_bins.astype(float)

    mi_cap = mi_binary_continuous(correct, capability, n_bins=3)
    cond_vals = []
    for b in np.unique(cap_bins[~np.isnan(cap_bins)]):
        mask = cap_bins == b
        if mask.sum() < 50:
            continue
        mi_b = mi_binary_continuous(correct[mask], reg_enc[mask], n_bins=2)
        cond_vals.append((mi_b, mask.sum() / len(df)))
    mi_cond = sum(m*w for m,w in cond_vals) if cond_vals else 0.0
    eil = mi_cond / mi_cap if mi_cap > 1e-9 else 0.0

    # Formal vs informal accuracy
    f_acc = float(df[df["register"]=="formal"]["correct"].mean())
    i_acc = float(df[df["register"]=="informal"]["correct"].mean())

    return {
        "EIL":              round(eil, 4),
        "formal_accuracy":  round(f_acc, 4),
        "informal_accuracy": round(i_acc, 4),
        "register_gap":     round(i_acc - f_acc, 4),
        "note": "EIL=0 expected with <20 models. Use register_gap as primary metric.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2PL IRT
# ══════════════════════════════════════════════════════════════════════════════

def fit_2pl(response_matrix):
    n_items, n_models = response_matrix.shape
    theta = response_matrix.mean(axis=0)
    a_vals, b_vals = np.zeros(n_items), np.zeros(n_items)

    for i in range(n_items):
        y = response_matrix[i, :]
        if y.sum() == 0 or y.sum() == n_models:
            a_vals[i] = 0.0
            b_vals[i] = 3.0 if y.sum() == 0 else -3.0
            continue

        def nll(params):
            a, b = params
            if a <= 0:
                return 1e9
            p = np.clip(expit(a * (theta - b)), 1e-9, 1-1e-9)
            return -(y * np.log(p) + (1-y) * np.log(1-p)).sum()

        res = minimize(nll, [1.0, 0.0], method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3})
        a_vals[i] = max(res.x[0], 0.0)
        b_vals[i] = res.x[1]

    return a_vals, b_vals


def compute_irt(df):
    models = sorted(df["model"].unique())
    results = []

    for register in ["formal", "informal"]:
        subset      = df[df["register"] == register]
        sent_ids    = sorted(subset["sentence_id"].unique())
        id_idx      = {sid: i for i, sid in enumerate(sent_ids)}
        model_idx   = {m: j for j, m in enumerate(models)}
        matrix      = np.zeros((len(sent_ids), len(models)))

        for _, row in subset.iterrows():
            i = id_idx.get(row["sentence_id"])
            j = model_idx.get(row["model"])
            if i is not None and j is not None:
                matrix[i, j] = row["correct"]

        print(f"  Fitting IRT: {register} ({len(sent_ids)} items × {len(models)} models)...")
        a_vals, b_vals = fit_2pl(matrix)

        for i, sid in enumerate(sent_ids):
            results.append({
                "sentence_id":    sid,
                "register":       register,
                "discrimination": round(a_vals[i], 4),
                "difficulty":     round(b_vals[i], 4),
            })

    return pd.DataFrame(results)


def construct_shift(irt_df):
    formal   = irt_df[irt_df["register"]=="formal"][["sentence_id","discrimination"]].rename(
        columns={"discrimination":"a_formal"})
    informal = irt_df[irt_df["register"]=="informal"][["sentence_id","discrimination"]].rename(
        columns={"discrimination":"a_informal"})
    merged   = formal.merge(informal, on="sentence_id", how="inner")

    print(f"\n  Mean discrimination (formal):   {irt_df[irt_df['register']=='formal']['discrimination'].mean():.4f}")
    print(f"  Mean discrimination (informal): {irt_df[irt_df['register']=='informal']['discrimination'].mean():.4f}")

    if len(merged) < 10:
        return {"r": None, "n": len(merged),
                "interpretation": "insufficient overlap — registers from different sources"}

    r, p = pearsonr(merged["a_formal"], merged["a_informal"])
    interp = ("Same construct, fixable" if r > 0.8
              else "Moderate shift" if r > 0.5
              else "Strong construct shift — not fixable by recalibration")
    print(f"  Pearson r (a_formal vs a_informal) = {r:.4f}  ({interp})")
    return {"r": round(r, 4), "p": round(p, 4),
            "n": len(merged), "interpretation": interp}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Step 3: EIL + IRT (Supplementary)")
    print("NOTE: EIL requires 20+ models for reliable estimation.")
    print("      With 8 models, EIL=0 is expected. See register_gap instead.\n")

    df = load_predictions(PREDICTIONS_PATH)

    # Balance dataset
    formal_ids   = df[df["register"]=="formal"]["sentence_id"].unique()
    informal_ids = df[df["register"]=="informal"]["sentence_id"].unique()
    n            = min(len(formal_ids), len(informal_ids))
    rng          = np.random.default_rng(42)
    sampled      = set(rng.choice(formal_ids, n, replace=False)) | set(informal_ids)
    df_bal       = df[df["sentence_id"].isin(sampled)]
    print(f"Balanced: {n} formal + {n} informal = {len(sampled)} sentences")

    cap_df = get_capability(df_bal, balanced=True)
    print("\nModel capabilities (balanced):")
    print(cap_df[["model","capability","capability_bin"]].to_string(index=False))

    eil_result = compute_eil(df_bal, cap_df)
    print(f"\nEIL = {eil_result['EIL']:.4f}")
    print(f"Register gap = {eil_result['register_gap']:+.4f} "
          f"(informal={eil_result['informal_accuracy']:.4f}, "
          f"formal={eil_result['formal_accuracy']:.4f})")

    # IRT
    print("\nFitting 2PL IRT...")
    irt_df = compute_irt(df_bal)
    irt_df.to_csv(IRT_RESULTS_PATH, index=False)
    print(f"Saved → {IRT_RESULTS_PATH}")

    cs = construct_shift(irt_df)
    save_json(cs, CONSTRUCT_SHIFT_PATH)

    save_json({"eil": eil_result, "construct_shift": cs}, EIL_RESULTS_PATH)
    print("\n✅ Step 3 complete.")


if __name__ == "__main__":
    main()
