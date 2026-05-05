"""
s4_sensitivity.py — Step 4: Sensitivity Curves.
FIXED: matplotlib legend bug in plot_sensitivity()
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kendalltau
from config import (PREDICTIONS_PATH, SENSITIVITY_PATH, SENSITIVITY_BP_PATH,
                    FIG_DIR, SENSITIVITY_EVAL_SIZE, SENSITIVITY_N_SEEDS, SENSITIVITY_RATIOS)
from utils import load_predictions, macro_f1


def compute_ranking(subset):
    rows = []
    for model, g in subset.groupby("model"):
        rows.append({"model": model,
                     "macro_F1": round(macro_f1(g["true_label"], g["predicted_label"]), 4)})
    r = pd.DataFrame(rows)
    r["rank"] = r["macro_F1"].rank(ascending=False, method="min").astype(int)
    return r.sort_values("rank")


def run_sensitivity(df):
    formal_ids   = df[df["register"]=="formal"]["sentence_id"].unique()
    informal_ids = df[df["register"]=="informal"]["sentence_id"].unique()
    models       = sorted(df["model"].unique())
    rows         = []

    print(f"Formal pool: {len(formal_ids):,}  |  Informal pool: {len(informal_ids):,}")

    rng      = np.random.default_rng(0)
    base_ids = rng.choice(informal_ids,
                          size=min(SENSITIVITY_EVAL_SIZE, len(informal_ids)),
                          replace=len(informal_ids) < SENSITIVITY_EVAL_SIZE)
    base_sub = df[df["sentence_id"].isin(base_ids)]
    base_r   = compute_ranking(base_sub)
    base_d   = dict(zip(base_r["model"], base_r["rank"]))
    base_top = base_r.iloc[0]["model"]
    print(f"Baseline top model (0% formal): {base_top}")

    for ratio in SENSITIVITY_RATIOS:
        for seed in range(SENSITIVITY_N_SEEDS):
            rng        = np.random.default_rng(seed)
            n_formal   = int(SENSITIVITY_EVAL_SIZE * ratio)
            n_informal = SENSITIVITY_EVAL_SIZE - n_formal

            f_ids = rng.choice(formal_ids, n_formal,
                               replace=len(formal_ids) < n_formal) if n_formal > 0 else []
            i_ids = rng.choice(informal_ids, n_informal,
                               replace=len(informal_ids) < n_informal) if n_informal > 0 else []
            sampled  = set(list(f_ids)) | set(list(i_ids))
            subset   = df[df["sentence_id"].isin(sampled)]
            ranking  = compute_ranking(subset)
            curr_d   = dict(zip(ranking["model"], ranking["rank"]))
            common   = [m for m in models if m in base_d and m in curr_d]
            tau, tau_p = kendalltau([base_d[m] for m in common],
                                    [curr_d[m]  for m in common])
            rows.append({"formal_ratio": ratio, "seed": seed,
                         "kendall_tau": round(tau, 4), "tau_p": round(tau_p, 4),
                         "top_model": ranking.iloc[0]["model"],
                         "n_sentences": len(sampled)})

        mean_tau = np.mean([r["kendall_tau"] for r in rows
                            if r["formal_ratio"] == ratio])
        print(f"  ratio={ratio:.0%}  mean_τ={mean_tau:.3f}")

    return pd.DataFrame(rows)


def find_breakpoints(curves_df):
    base_top = curves_df[curves_df["formal_ratio"]==0.0]["top_model"].mode()[0]
    summary  = curves_df.groupby("formal_ratio").agg(
        mean_tau=("kendall_tau","mean"),
        top_mode=("top_model", lambda x: x.mode()[0])
    ).reset_index()
    bp_tau = summary[summary["mean_tau"] < 0.5]["formal_ratio"].min()
    bp_top = summary[summary["top_mode"] != base_top]["formal_ratio"].min()
    print(f"\nBaseline top: {base_top}")
    print(f"τ < 0.5 at: {bp_tau:.0%}" if not pd.isna(bp_tau) else "τ never drops below 0.5")
    print(f"Top changes: {bp_top:.0%}" if not pd.isna(bp_top) else "Top model never changes")
    return {"baseline_top": base_top,
            "breakpoint_tau": float(bp_tau) if not pd.isna(bp_tau) else None,
            "breakpoint_top": float(bp_top) if not pd.isna(bp_top) else None}


def plot_sensitivity(curves_df, breakpoints):
    summary = curves_df.groupby("formal_ratio").agg(
        mean_tau=("kendall_tau","mean"),
        std_tau=("kendall_tau","std"),
    ).reset_index()

    mods    = curves_df["top_model"].unique()
    cmap    = dict(zip(mods, sns.color_palette("tab10", len(mods))))
    N_SEEDS = curves_df["seed"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = summary["formal_ratio"] * 100

    ax = axes[0]
    ax.plot(x, summary["mean_tau"], "o-", color="steelblue", lw=2, label="Mean Kendall's τ")
    ax.fill_between(x,
                    summary["mean_tau"] - summary["std_tau"],
                    summary["mean_tau"] + summary["std_tau"],
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.axhline(0.5, color="red", ls="--", lw=1.5, label="τ=0.5 threshold")
    bp_top = breakpoints.get("breakpoint_top")
    if bp_top is not None:
        ax.axvline(bp_top*100, color="orange", ls="-.", lw=1.5,
                   label=f"Top model changes: {bp_top:.0%}")
    ax.set_xlabel("Formal text proportion (%)"); ax.set_ylabel("Kendall's τ")
    ax.set_title("Ranking Stability vs Formal Proportion")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes[1]
    for seed in range(N_SEEDS):
        seed_df = curves_df[curves_df["seed"] == seed]
        for _, row in seed_df.iterrows():
            ax2.scatter(row["formal_ratio"]*100, seed,
                        color=cmap.get(row["top_model"], "gray"), s=60, zorder=3)
    # FIXED: pass handles and labels separately
    handles = [mpatches.Patch(color=cmap[m]) for m in mods]
    ax2.legend(handles, list(mods), fontsize=7, title="Top Model", loc="upper right")
    ax2.set_xlabel("Formal text proportion (%)"); ax2.set_ylabel("Seed")
    ax2.set_title("Top Model by Ratio and Seed")
    ax2.set_yticks(range(N_SEEDS)); ax2.grid(alpha=0.3)

    plt.suptitle("Sensitivity Analysis: Register Mix vs Ranking Stability", y=1.02)
    path = FIG_DIR / "figure4_sensitivity_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def main():
    print("Step 4: Sensitivity Curves")
    df = load_predictions(PREDICTIONS_PATH)
    print(f"Loaded {len(df):,} rows")
    curves_df = run_sensitivity(df)
    curves_df.to_csv(SENSITIVITY_PATH, index=False)
    print(f"Saved → {SENSITIVITY_PATH}")
    bps = find_breakpoints(curves_df)
    with open(SENSITIVITY_BP_PATH, "w") as f:
        json.dump(bps, f, indent=2)
    print(f"Saved → {SENSITIVITY_BP_PATH}")
    plot_sensitivity(curves_df, bps)
    print("\n✅ Step 4 complete.")


if __name__ == "__main__":
    main()
