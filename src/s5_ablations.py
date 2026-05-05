"""
s5_ablations.py — Step 5: Ablations + Failure Analysis.
FIXED: handles missing 'text' column gracefully.
Skips length/readability ablations if text not present.
Falls back to source-based ablation only.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY = True
except OSError:
    SPACY = False

try:
    import textstat
    TEXTSTAT = True
except ImportError:
    TEXTSTAT = False

from config import (PREDICTIONS_PATH, GRD_RESULTS_PATH, ABLATION_PATH,
                    FAILURE_PATH, FAILURE_REVIEW_PATH, FIG_DIR,
                    N_HARD_SENTENCES, N_MANUAL_REVIEW)
from utils import load_predictions
from s2_grd import compute_grd


# ── Text features (only if text column exists) ────────────────────────────────

def add_text_features(df):
    if "text" not in df.columns:
        print("  WARNING: 'text' column not found — skipping length/readability features")
        return df, False

    if not TEXTSTAT:
        print("  WARNING: textstat not installed — skipping readability features")
        return df, False

    print("  Computing text features...")
    unique = df.drop_duplicates("sentence_id")[["sentence_id","text"]].copy()
    unique["word_count"]      = unique["text"].apply(lambda x: len(str(x).split()))
    unique["flesch_kincaid"]  = unique["text"].apply(
        lambda x: textstat.flesch_kincaid_grade(str(x)))
    unique["length_bin"]      = pd.qcut(unique["word_count"], q=3,
                                        labels=["short","medium","long"], duplicates="drop")
    unique["readability_bin"] = pd.qcut(unique["flesch_kincaid"], q=3,
                                        labels=["easy","medium","hard"], duplicates="drop")
    df = df.merge(unique[["sentence_id","word_count","flesch_kincaid",
                           "length_bin","readability_bin"]], on="sentence_id", how="left")
    return df, True


def grd_simple(df):
    if df["register"].nunique() < 2:
        return 0.0
    r = compute_grd(df, group_col="register", ref_group="informal", cmp_group="formal")
    return r["GRD_global"]


# ── Ablation by bin ───────────────────────────────────────────────────────────

def ablation_by_bin(df, bin_col, baseline_grd, label):
    if bin_col not in df.columns:
        print(f"  Skipping {label} ablation — column '{bin_col}' not found")
        return []

    rows = []
    for b in df[bin_col].dropna().unique():
        subset = df[df[bin_col] == b]
        if subset["register"].nunique() < 2:
            continue
        grd = grd_simple(subset)
        ret = round(grd / baseline_grd * 100, 1) if baseline_grd > 0 else 0
        rows.append({"ablation": label, "condition": str(b),
                     "GRD": round(grd, 4), "baseline_GRD": baseline_grd,
                     "retention_pct": ret})
        print(f"  {str(b):>12}: GRD={grd:.4f}  retention={ret:.0f}%")

    # Matched version
    matched_dfs = []
    for b in df[bin_col].dropna().unique():
        bin_df = df[df[bin_col] == b]
        f_ids  = bin_df[bin_df["register"]=="formal"]["sentence_id"].unique()
        i_ids  = bin_df[bin_df["register"]=="informal"]["sentence_id"].unique()
        n      = min(len(f_ids), len(i_ids))
        if n < 10:
            continue
        matched_dfs.append(bin_df[bin_df["sentence_id"].isin(
            list(np.random.choice(f_ids, n, replace=False)) +
            list(np.random.choice(i_ids, n, replace=False))
        )])

    if matched_dfs:
        matched = pd.concat(matched_dfs)
        grd_m   = grd_simple(matched)
        ret     = round(grd_m / baseline_grd * 100, 1) if baseline_grd > 0 else 0
        flag    = "✅ REAL — register effect persists" if ret >= 70 else "⚠️  CONFOUNDED"
        rows.append({"ablation": label, "condition": f"{label}-matched",
                     "GRD": round(grd_m, 4), "baseline_GRD": baseline_grd,
                     "retention_pct": ret})
        print(f"  {label}-matched: GRD={grd_m:.4f}  {flag}")

    return rows


# ── Source ablation (works without text column) ───────────────────────────────

def ablation_source(df, baseline_grd):
    print("  Source ablation:")
    rows = []
    for source, g in df.groupby("source"):
        if g["register"].nunique() < 2:
            continue
        grd = grd_simple(g)
        ret = round(grd / baseline_grd * 100, 1) if baseline_grd > 0 else 0
        rows.append({"ablation": "source", "condition": str(source),
                     "GRD": round(grd, 4), "baseline_GRD": baseline_grd,
                     "retention_pct": ret})
        print(f"  {str(source)[:40]:>42}: GRD={grd:.4f}")
    return rows


# ── Failure analysis ──────────────────────────────────────────────────────────

LEGAL_EUPHEMISMS = [
    "inharmonious","infiltration","undesirable elements","neighborhood character",
    "community stability","inner-city","urban renewal","blight","security risk",
]
CODED_TERMS = [
    "negro","colored","mulatto","asiatic","mongolian","alien race",
    "miscegenation","white only","jim crow","grandfather clause","racial integrity",
]
HEDGING = [" may "," might "," generally "," typically "," subject to ",
           " at the discretion "," as appropriate "]
PROCEDURAL = [
    "pursuant to","in accordance with","as provided by","per section",
    "under subsection","hereinafter","shall be","eligible persons",
]
EXPLICIT_DEMO = [
    "race","racial","ethnic","religion","gender","sex","national origin",
    "disability","age","sexual orientation","immigrant","citizen","refugee",
]


def detect_features(text):
    t = text.lower()
    feats = {
        "passive_voice":        False,
        "legal_euphemism":      any(e in t for e in LEGAL_EUPHEMISMS),
        "coded_historical":     any(e in t for e in CODED_TERMS),
        "hedging":              any(h in t for h in HEDGING),
        "procedural_framing":   any(p in t for p in PROCEDURAL),
        "explicit_demographic": any(e in t for e in EXPLICIT_DEMO),
    }
    if SPACY:
        doc = nlp(text)
        feats["passive_voice"] = any(
            tok.dep_ == "auxpass" or
            (tok.dep_ == "nsubjpass" and tok.head.pos_ == "VERB")
            for tok in doc)
    return feats


def primary_category(row):
    if row.get("coded_historical"):    return "Coded Historical Language"
    if row.get("legal_euphemism"):     return "Legal Euphemism"
    if row.get("passive_voice"):       return "Passive Voice"
    if row.get("procedural_framing"):  return "Procedural Framing"
    if row.get("hedging"):             return "Hedging / Softening"
    if row.get("explicit_demographic"):return "Explicit Demographic (Missed)"
    return "Other / Unclear"


def run_failure_analysis(df):
    print(f"\nFinding {N_HARD_SENTENCES} hardest sentences...")
    error_rate = (df.groupby("sentence_id")["correct"]
                  .agg(["mean","count"]).reset_index()
                  .rename(columns={"mean":"accuracy","count":"n_models"}))
    error_rate["error_rate"] = 1 - error_rate["accuracy"]
    hard = error_rate.nlargest(N_HARD_SENTENCES, "error_rate")

    # Metadata columns that exist in df
    meta_cols = ["sentence_id","true_label","register","era","source","bias_type"]
    if "text" in df.columns:
        meta_cols.append("text")
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df.drop_duplicates("sentence_id")[meta_cols]
    hard = hard.merge(meta, on="sentence_id", how="left")

    print(f"  Mean error rate: {hard['error_rate'].mean():.3f}")
    print(f"  Register: {hard['register'].value_counts().to_dict()}")
    print(f"  Era: {hard['era'].value_counts().to_dict()}")
    print(f"  Bias type: {hard['bias_type'].value_counts().to_dict()}")

    # Linguistic feature detection only if text is available
    if "text" in hard.columns:
        print("  Detecting linguistic features...")
        feat_rows = []
        for _, row in hard.iterrows():
            f = detect_features(str(row.get("text","")))
            f["sentence_id"] = row["sentence_id"]
            feat_rows.append(f)
        feat_df = pd.DataFrame(feat_rows)
        hard    = hard.merge(feat_df, on="sentence_id", how="left")
        hard["failure_category"] = hard.apply(primary_category, axis=1)
        print("\n  Failure categories:")
        for cat, cnt in hard["failure_category"].value_counts().items():
            print(f"    {cat:<40} {cnt:>4}  ({cnt/len(hard)*100:.1f}%)")
    else:
        hard["failure_category"] = "Unknown (no text column)"
        print("  NOTE: text column not available — skipping linguistic feature detection")
        print("        P7 should look up sentence_ids in original dataset for manual review")

    # Manual review sample
    group_cols = ["register","bias_type"] if "failure_category" not in hard.columns \
                 else ["register","failure_category"]
    review = hard.groupby(group_cols, group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, N_MANUAL_REVIEW // hard[group_cols[-1]].nunique())))
    ).head(N_MANUAL_REVIEW)

    return hard, review


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_ablations(abl_df, baseline_grd):
    if len(abl_df) == 0:
        print("  No ablation data to plot")
        return

    ablation_types = abl_df["ablation"].unique()
    n = len(ablation_types)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5*max(n,1), 5))
    if n == 1:
        axes = [axes]

    for ax, abl in zip(axes, ablation_types):
        sub    = abl_df[abl_df["ablation"] == abl]
        colors = ["steelblue" if "matched" not in c else "coral"
                  for c in sub["condition"]]
        bars   = ax.barh(sub["condition"], sub["GRD"], color=colors, alpha=0.8)
        ax.axvline(baseline_grd, color="red", ls="--", lw=1.5,
                   label=f"Baseline={baseline_grd:.3f}")
        ax.axvline(baseline_grd*0.7, color="orange", ls=":", lw=1,
                   label="70% threshold")
        for bar, grd in zip(bars, sub["GRD"]):
            ax.text(grd+0.002, bar.get_y()+bar.get_height()/2,
                    f"{grd:.3f}", va="center", fontsize=9)
        ax.set_xlabel("GRD"); ax.set_title(f"Ablation: {abl}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    plt.suptitle("Ablations: Does GRD Persist After Controlling Confounds?",
                 fontsize=12, y=1.02)
    path = FIG_DIR / "figure5_ablations.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_failure(fail_df):
    if "failure_category" not in fail_df.columns or \
       fail_df["failure_category"].eq("Unknown (no text column)").all():
        # Simple register/bias_type breakdown instead
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fail_df["register"].value_counts().plot(kind="bar", ax=axes[0],
            color=["#C0392B","#2980B9"], alpha=0.85)
        axes[0].set_title("Hard Sentences by Register")
        axes[0].set_ylabel("Count"); axes[0].grid(alpha=0.3, axis="y")
        fail_df["bias_type"].value_counts().plot(kind="bar", ax=axes[1],
            color=["#E74C3C","#27AE60","#8E44AD"], alpha=0.85)
        axes[1].set_title("Hard Sentences by Bias Type")
        axes[1].set_ylabel("Count"); axes[1].grid(alpha=0.3, axis="y")
        plt.suptitle("Failure Analysis — Hard Sentences (Error Rate = 1.0)", fontsize=12)
    else:
        cats    = fail_df["failure_category"].unique()
        palette = dict(zip(cats, sns.color_palette("tab10", len(cats))))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        counts = fail_df["failure_category"].value_counts()
        axes[0].pie(counts.values, labels=counts.index,
                    colors=[palette[c] for c in counts.index],
                    autopct="%1.1f%%", startangle=90, textprops={"fontsize":9})
        axes[0].set_title("Failure Mode Distribution")
        cross = pd.crosstab(fail_df["register"], fail_df["failure_category"],
                            normalize="index") * 100
        cross.plot(kind="bar", stacked=True, ax=axes[1],
                   color=[palette.get(c,"gray") for c in cross.columns],
                   edgecolor="white", lw=0.5)
        axes[1].set_xlabel("Register"); axes[1].set_ylabel("% of hard sentences")
        axes[1].set_title("Failure Mode by Register")
        axes[1].legend(bbox_to_anchor=(1.05,1), fontsize=8)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].grid(alpha=0.3, axis="y")
        plt.suptitle("Failure Analysis: Why Do Models Fail?", fontsize=12, y=1.02)

    path = FIG_DIR / "figure6_failure_taxonomy.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Step 5: Ablations + Failure Analysis")

    df = load_predictions(PREDICTIONS_PATH)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Load baseline GRD
    with open(GRD_RESULTS_PATH) as f:
        grd_data = json.load(f)
    baseline_grd = grd_data["register"]["GRD_global"]
    print(f"Baseline GRD = {baseline_grd:.4f}")

    all_rows = []

    # Text-based ablations (only if text column exists)
    df, has_text = add_text_features(df)
    if has_text:
        print("\nAblation 1: Length")
        all_rows += ablation_by_bin(df, "length_bin", baseline_grd, "length")
        print("\nAblation 2: Readability")
        all_rows += ablation_by_bin(df, "readability_bin", baseline_grd, "complexity")
    else:
        print("\nSkipping length + readability ablations (no text column)")
        print("NOTE: To enable these, add 'text' column to predictions file")

    # Source ablation (always runs)
    print("\nAblation: Source")
    all_rows += ablation_source(df, baseline_grd)

    abl_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["ablation","condition","GRD","baseline_GRD","retention_pct"])
    abl_df.to_csv(ABLATION_PATH, index=False)
    print(f"\nSaved → {ABLATION_PATH}")

    # Failure analysis
    hard_df, review_df = run_failure_analysis(df)
    hard_df.to_csv(FAILURE_PATH, index=False)
    review_df.to_csv(FAILURE_REVIEW_PATH, index=False)
    print(f"Saved → {FAILURE_PATH}")
    print(f"Saved → {FAILURE_REVIEW_PATH}")
    print(f"\n→ P7: review {FAILURE_REVIEW_PATH} ({len(review_df)} sentences)")

    # Figures
    plot_ablations(abl_df, baseline_grd)
    plot_failure(hard_df)

    print("\n✅ Step 5 complete.")


if __name__ == "__main__":
    main()
