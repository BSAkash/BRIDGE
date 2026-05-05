"""
s7_figures.py — Step 7: All paper figures.
UPDATED: uses actual results from completed experiments.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from config import (RANKINGS_INFORMAL_PATH, RANKINGS_FORMAL_PATH,
                    RANKING_FLIP_PATH, GRD_RESULTS_PATH, EXTERNAL_GRD_PATH,
                    FIG_DIR, RRS_PATH)

sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
COLORS = {"formal": "#C0392B", "informal": "#2980B9",
          "harmful": "#E74C3C", "harmless": "#27AE60", "antibias": "#8E44AD"}


# ── Figure 1: Ranking Flip ─────────────────────────────────────────────────────

def figure1_ranking_flip():
    if not (RANKING_FLIP_PATH.exists() and
            RANKINGS_INFORMAL_PATH.exists() and
            RANKINGS_FORMAL_PATH.exists()):
        print("Skipping Figure 1 — missing ranking files"); return

    r_inf  = pd.read_csv(RANKINGS_INFORMAL_PATH)
    r_frm  = pd.read_csv(RANKINGS_FORMAL_PATH)
    flip   = pd.read_csv(RANKING_FLIP_PATH)
    models = flip["model"].tolist()
    x      = np.arange(len(models))
    w      = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: F1 bars
    ax = axes[0]
    merged = flip.merge(r_inf[["model","macro_F1"]], on="model").rename(
        columns={"macro_F1":"f1_informal"})
    merged = merged.merge(r_frm[["model","macro_F1"]], on="model").rename(
        columns={"macro_F1":"f1_formal"})
    ax.bar(x - w/2, merged["f1_informal"], w, label="Informal register",
           color=COLORS["informal"], alpha=0.85)
    ax.bar(x + w/2, merged["f1_formal"],   w, label="Formal register",
           color=COLORS["formal"],   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split()[0] for m in models], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1"); ax.set_title("Model Performance by Register", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1)

    # Right: rank lines
    ax2 = axes[1]
    colors_line = sns.color_palette("tab10", len(models))
    for idx, (_, row) in enumerate(flip.iterrows()):
        ax2.plot([0, 1], [row["rank_informal"], row["rank_formal"]],
                 "o-", color=colors_line[idx], alpha=0.8, lw=2, ms=8)
        ax2.annotate(row["model"].split()[0],
                     xy=(0, row["rank_informal"]),
                     xytext=(-0.15, row["rank_informal"]),
                     fontsize=8, ha="right", va="center")
        ax2.annotate(row["model"].split()[0],
                     xy=(1, row["rank_formal"]),
                     xytext=(1.02, row["rank_formal"]),
                     fontsize=8, ha="left", va="center")

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Informal\nRegister", "Formal\nRegister"], fontsize=11)
    ax2.set_ylabel("Rank (1=best)"); ax2.invert_yaxis()
    ax2.set_title("Model Ranking Flip Across Registers", fontsize=12)
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle("Figure 1: Register Dependency in LLM Bias Detection Rankings",
                 fontsize=13, fontweight="bold", y=1.02)
    path = FIG_DIR / "figure1_ranking_flip.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 2: GRD Heatmap ─────────────────────────────────────────────────────

def figure2_grd_heatmap():
    if not GRD_RESULTS_PATH.exists():
        print("Skipping Figure 2 — grd_results.json not found"); return

    with open(GRD_RESULTS_PATH) as f:
        grd_data = json.load(f)

    groupings = [k for k in grd_data if k not in ("ccd",) and
                 isinstance(grd_data[k], dict) and "per_class" in grd_data[k]]
    if not groupings:
        print("Skipping Figure 2 — no per_class data found"); return

    classes = sorted(grd_data[groupings[0]]["per_class"].keys())
    matrix  = []
    for g in groupings:
        matrix.append([grd_data[g]["per_class"].get(c,{}).get("GRD",0) for c in classes])
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(6, len(classes)*1.5), max(4, len(groupings)*1.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.55, aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(range(len(groupings))); ax.set_yticklabels(groupings, fontsize=11)
    for i in range(len(groupings)):
        for j in range(len(classes)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, color="black" if matrix[i,j] < 0.35 else "white")
    plt.colorbar(im, ax=ax, label="GRD (higher = more register sensitivity)")
    ax.set_title("Figure 2: GRD Across Grouping Variables and Bias Classes",
                 fontsize=12, fontweight="bold")
    path = FIG_DIR / "figure2_grd_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ── Figure 3: Polarity Flip ───────────────────────────────────────────────────

def figure3_polarity_flip():
    if not (GRD_RESULTS_PATH.exists() and RRS_PATH.exists()):
        print("Skipping Figure 3 — missing files"); return

    rrs_df = pd.read_csv(RRS_PATH)
    with open(GRD_RESULTS_PATH) as f:
        grd_data = json.load(f)

    classes = sorted(rrs_df["class"].unique())
    models  = sorted(rrs_df["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: recall lines per class × register
    ax  = axes[0]
    x   = np.arange(len(models))
    for c in classes:
        sub_f = rrs_df[rrs_df["class"]==c].set_index("model")["recall_cmp"]
        sub_i = rrs_df[rrs_df["class"]==c].set_index("model")["recall_ref"]
        f_v = [sub_f.get(m, 0) for m in models]
        i_v = [sub_i.get(m, 0) for m in models]
        ax.plot(x, i_v, "o--", color=COLORS.get(c,"gray"), alpha=0.6, lw=1.5, ms=6,
                label=f"{c} (informal)")
        ax.plot(x, f_v, "s-",  color=COLORS.get(c,"gray"), alpha=1.0, lw=2,   ms=8,
                label=f"{c} (formal)")

    ax.set_xticks(x)
    ax.set_xticklabels([m.split()[0] for m in models], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
    ax.set_title("Recall by Class and Register", fontsize=12)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # Right: DC bar chart
    ax2 = axes[1]
    reg_classes = sorted(grd_data["register"]["per_class"].keys())
    dc_vals     = [grd_data["register"]["per_class"][c]["DC"] for c in reg_classes]
    grd_signed  = [grd_data["register"]["per_class"][c]["GRD_signed"] for c in reg_classes]
    bar_colors  = [COLORS.get(c,"steelblue") for c in reg_classes]

    bars = ax2.barh(reg_classes, dc_vals, color=bar_colors, alpha=0.85)
    ax2.axvline(0.8, color="red",     ls="--", lw=1.5, label="DC=0.8 threshold")
    ax2.axvline(1.0, color="darkred", ls="-",  lw=2,   label="DC=1.0 (all models)")

    for bar, dc, gs in zip(bars, dc_vals, grd_signed):
        direction = f"↑ informal > formal  gap={gs:+.3f}" if gs > 0 \
                    else f"↓ formal > informal  gap={gs:+.3f}"
        ax2.text(dc + 0.01, bar.get_y() + bar.get_height()/2,
                 f"DC={dc:.2f}  {direction}", va="center", fontsize=9)

    ax2.set_xlim(0, 1.35)
    ax2.set_xlabel("Directional Consistency (DC)")
    ax2.set_title("DC per Class + Register Polarity Flip", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="x")

    rpf = grd_data["register"]["RPF"]
    ax2.text(0.97, 0.05,
             f"RPF = {rpf}\n({'Polarity Flip' if rpf else 'No flip'})",
             transform=ax2.transAxes, fontsize=10, color="red", ha="right",
             bbox=dict(boxstyle="round", fc="lightyellow", ec="red"))

    plt.suptitle("Figure 3: Register Polarity Flip — DC=1.0 for Harmful and Harmless",
                 fontsize=13, fontweight="bold", y=1.02)
    path = FIG_DIR / "figure3_polarity_flip.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Figure 7: External Validation ─────────────────────────────────────────────

def figure7_external_validation():
    if not (GRD_RESULTS_PATH.exists() and EXTERNAL_GRD_PATH.exists()):
        print("Skipping Figure 7 — missing files"); return

    with open(GRD_RESULTS_PATH)  as f: internal = json.load(f)
    with open(EXTERNAL_GRD_PATH) as f: external = json.load(f)

    int_pc  = internal.get("register",{}).get("per_class",{})
    ext_pc  = external.get("per_class",{})
    classes = sorted(set(int_pc.keys()) & set(ext_pc.keys()))
    if not classes:
        print("Skipping Figure 7 — no matching classes"); return

    int_grd = [int_pc[c]["GRD"] for c in classes]
    ext_grd = [ext_pc[c]["GRD"] for c in classes]
    int_dc  = [int_pc[c]["DC"]  for c in classes]
    ext_dc  = [ext_pc[c]["DC"]  for c in classes]

    x     = np.arange(len(classes))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GRD comparison
    ax = axes[0]
    ax.bar(x - width/2, int_grd, width, label="Internal (RBB)", color="steelblue", alpha=0.85)
    ax.bar(x + width/2, ext_grd, width, label="External (BABE+MHS)", color="coral", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("GRD"); ax.set_title("GRD Replication: Internal vs External")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.text(0.02, 0.95,
            f"Internal:  GRD={internal['register']['GRD_global']:.3f}, "
            f"RPF={internal['register']['RPF']}\n"
            f"External:  GRD={external.get('GRD_global',0):.3f}, "
            f"RPF={external.get('RPF',0)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="lightyellow"))

    # DC comparison
    ax2 = axes[1]
    ax2.bar(x - width/2, int_dc, width, label="Internal DC", color="steelblue", alpha=0.85)
    ax2.bar(x + width/2, ext_dc, width, label="External DC", color="coral", alpha=0.85)
    ax2.axhline(1.0, color="darkred", ls="--", lw=1.5, label="DC=1.0")
    ax2.axhline(0.8, color="red",     ls=":",  lw=1,   label="DC=0.8 threshold")
    ax2.set_xticks(x); ax2.set_xticklabels(classes, fontsize=12)
    ax2.set_ylabel("Directional Consistency (DC)")
    ax2.set_title("DC Replication: Internal vs External")
    ax2.set_ylim(0, 1.15); ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    plt.suptitle("Figure 7: External Validation — Register Polarity Flip Replicates",
                 fontsize=13, fontweight="bold", y=1.02)
    path = FIG_DIR / "figure7_external_validation.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Step 7: Generate All Figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    figure1_ranking_flip()
    figure2_grd_heatmap()
    figure3_polarity_flip()
    print("Note: Figures 4, 5, 6 generated by Steps 4 and 5")
    figure7_external_validation()

    print(f"\n✅ Step 7 complete. Figures in {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
