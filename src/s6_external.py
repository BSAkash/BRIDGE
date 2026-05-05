"""
s6_external.py — Step 6: External Validation.
UPDATED: handles per-model files + model name normalization.

Two modes:
  A) Merge existing per-model files (if you already ran inference)
  B) Run inference from scratch via OpenRouter

Input:  Per-model JSONL files  OR  data/external_combined.jsonl + OpenRouter key
Output: data/external_predictions_merged.jsonl
        results/external_grd_results.json
"""

import os
import time
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from config import (EXTERNAL_COMBINED_PATH, EXTERNAL_PREDICTIONS_PATH,
                    EXTERNAL_GRD_PATH, MODEL_REGISTRY, EXTERNAL_PROMPT,
                    MHS_HARMFUL_THRESHOLD, MHS_HARMLESS_THRESHOLD, DATA_DIR)
from utils import save_json
from s2_grd import compute_grd, print_grd_report

# Merged output path
MERGED_PATH = DATA_DIR / "external_predictions_merged.jsonl"

# ── Model name normalization ───────────────────────────────────────────────────
# Maps any variant name → standardized name matching main predictions file
MODEL_NAME_MAP = {
    # OpenRouter API strings
    "openai/gpt-4o-mini":                    "GPT-5.4",
    "anthropic/claude-sonnet-4-5":           "Claude Sonnet 4.6",
    "google/gemini-pro-1.5":                 "Gemini 3.1 Pro Preview",
    "meta-llama/llama-3.1-70b-instruct":     "Llama 3.1 70B-Inst 4-bit",
    "meta-llama/llama-3.1-8b-instruct":      "Llama 3.1 8B-Inst",
    "qwen/qwen-2.5-72b-instruct":            "Qwen 2.5 72B Instruct 4-bit",
    "mistralai/mistral-7b-instruct-v0.3":    "Mistral 7B v0.3 Instruct",
    "google/gemma-2-9b-it":                  "Gemma 2 9B Instruct 4-bit",
    # Common variant names in per-model files
    "google/gemma-2-9b-it":                  "Gemma 2 9B Instruct 4-bit",
    "Llama 3.1 8B Instruct":                 "Llama 3.1 8B-Inst",
    "Qwen/Qwen2.5-72B-Instruct":             "Qwen 2.5 72B Instruct 4-bit",
    "Gemini 3.1 Pro":                        "Gemini 3.1 Pro Preview",
    "Claude 4.6 Sonnet":                     "Claude Sonnet 4.6",
}


# ══════════════════════════════════════════════════════════════════════════════
# MODE A: Merge existing per-model files
# ══════════════════════════════════════════════════════════════════════════════

def merge_existing_files(file_pattern: str = None) -> pd.DataFrame:
    """
    Merge per-model prediction files into one clean dataframe.
    Auto-detects files matching common patterns in DATA_DIR and /content/.
    """
    search_dirs = [str(DATA_DIR), "/content"]
    patterns    = ["*external_predictions*.jsonl", "*external*.jsonl"]
    found_files = []

    for d in search_dirs:
        for pat in patterns:
            found_files.extend(glob.glob(f"{d}/{pat}"))

    # Exclude already-merged files
    found_files = [f for f in found_files
                   if "merged" not in Path(f).name
                   and Path(f).exists()]
    found_files = list(set(found_files))

    if not found_files:
        return pd.DataFrame()

    print(f"Found {len(found_files)} per-model files:")
    for f in found_files:
        print(f"  {Path(f).name}")

    dfs = []
    for fpath in found_files:
        records = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df_tmp = pd.DataFrame(records)

        # Normalize model names
        if "model" in df_tmp.columns:
            df_tmp["model"] = df_tmp["model"].map(
                lambda x: MODEL_NAME_MAP.get(x, x))

        # Keep standard columns only
        keep = ["sentence_id","model","predicted_label","true_label","register","source"]
        df_tmp = df_tmp[[c for c in keep if c in df_tmp.columns]]
        dfs.append(df_tmp)
        print(f"  {Path(fpath).name}: {len(df_tmp):,} rows | "
              f"model={df_tmp['model'].unique()}")

    merged = pd.concat(dfs, ignore_index=True)

    # Remove unknowns
    before  = len(merged)
    merged  = merged[merged["predicted_label"] != "unknown"].copy()
    merged  = merged[merged["true_label"]      != "unknown"].copy()
    removed = before - len(merged)
    print(f"\nRemoved {removed:,} unknown rows ({removed/before*100:.1f}%)")
    print(f"Total: {len(merged):,} rows | {merged['model'].nunique()} models")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MODE B: Build dataset + run inference from scratch
# ══════════════════════════════════════════════════════════════════════════════

def build_combined_dataset() -> pd.DataFrame:
    from datasets import load_dataset

    print("Loading BABE (formal)...")
    babe = pd.concat([
        load_dataset("mediabiasgroup/BABE")["train"].to_pandas(),
        load_dataset("mediabiasgroup/BABE")["test"].to_pandas(),
    ], ignore_index=True)
    babe["true_label"]  = babe["label"].map({1: "harmful", 0: "harmless"})
    babe["register"]    = "formal"
    babe["source"]      = "BABE"
    babe["sentence_id"] = "babe_" + babe.index.astype(str)
    formal_df = babe[["sentence_id","text","true_label","register","source"]].dropna()
    print(f"  BABE: {len(formal_df)} rows")

    print("Loading Measuring Hate Speech (informal)...")
    ds  = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    mhs = ds.to_pandas()
    agg = (mhs.groupby("comment_id")
           .agg(text=("text","first"), mean_score=("hate_speech_score","mean"))
           .reset_index())

    def map_label(score):
        if score > MHS_HARMFUL_THRESHOLD:  return "harmful"
        if score < MHS_HARMLESS_THRESHOLD: return "harmless"
        return "ambiguous"

    agg["true_label"] = agg["mean_score"].apply(map_label)
    informal_df = agg[agg["true_label"] != "ambiguous"].copy()
    informal_df["register"]   = "informal"
    informal_df["source"]     = "MeasuringHateSpeech"
    informal_df = informal_df.rename(columns={"comment_id": "sentence_id"})
    informal_df = informal_df[["sentence_id","text","true_label","register","source"]]
    print(f"  MHS: {len(informal_df)} rows")

    combined = pd.concat([formal_df, informal_df], ignore_index=True)
    combined.to_json(EXTERNAL_COMBINED_PATH, orient="records", lines=True)
    print(f"Combined: {len(combined)} rows → {EXTERNAL_COMBINED_PATH}")
    return combined


def run_inference(combined: pd.DataFrame) -> pd.DataFrame:
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    done_pairs = set()
    if EXTERNAL_PREDICTIONS_PATH.exists():
        with open(EXTERNAL_PREDICTIONS_PATH) as f:
            for line in f:
                r = json.loads(line)
                done_pairs.add((str(r["sentence_id"]), r["model"]))
        print(f"Resuming — {len(done_pairs):,} done")

    for display_name, or_model in MODEL_REGISTRY.items():
        already = sum(1 for sid in combined["sentence_id"]
                      if (str(sid), display_name) in done_pairs)
        if already == len(combined):
            print(f"Skipping {display_name} — complete")
            continue

        print(f"\nRunning {display_name}...")
        model_rows = []

        for i, (_, row) in enumerate(combined.iterrows()):
            sid = str(row["sentence_id"])
            if (sid, display_name) in done_pairs:
                continue
            try:
                resp = client.chat.completions.create(
                    model=or_model,
                    messages=[{"role":"user",
                               "content": EXTERNAL_PROMPT.format(text=str(row["text"])[:500])}],
                    max_tokens=5, temperature=0,
                )
                raw   = resp.choices[0].message.content.lower().strip()
                label = "harmful" if "harmful" in raw else \
                        "harmless" if "harmless" in raw else "unknown"
            except Exception as e:
                print(f"  Error row {i}: {e}")
                label = "unknown"

            model_rows.append({
                "sentence_id": sid, "model": display_name,
                "predicted_label": label, "true_label": row["true_label"],
                "register": row["register"], "source": row["source"],
            })

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(combined)}")
                with open(EXTERNAL_PREDICTIONS_PATH, "a") as f:
                    for r2 in model_rows[-500:]:
                        f.write(json.dumps(r2) + "\n")
            time.sleep(0.03)

        with open(EXTERNAL_PREDICTIONS_PATH, "a") as f:
            remainder = len(model_rows) % 500
            start = len(model_rows) - remainder if remainder else 0
            for r2 in model_rows[start:]:
                f.write(json.dumps(r2) + "\n")

        correct = sum(1 for r2 in model_rows if r2["predicted_label"] == r2["true_label"])
        print(f"  Accuracy: {correct/len(model_rows):.4f}")

    return pd.read_json(EXTERNAL_PREDICTIONS_PATH, lines=True)


# ══════════════════════════════════════════════════════════════════════════════
# GRD
# ══════════════════════════════════════════════════════════════════════════════

def compute_external_grd(preds_df: pd.DataFrame) -> dict:
    preds_df = preds_df[preds_df["predicted_label"] != "unknown"].copy()
    preds_df["correct"] = (preds_df["predicted_label"] == preds_df["true_label"]).astype(int)

    print(f"\nExternal: {len(preds_df):,} rows | {preds_df['model'].nunique()} models")
    print("Per-model accuracy:")
    for model, g in preds_df.groupby("model"):
        print(f"  {model:<35} acc={g['correct'].mean():.4f}")

    results = compute_grd(preds_df, group_col="register",
                          ref_group="informal", cmp_group="formal")
    results.pop("raw_rrs", None)
    print_grd_report(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Step 6: External Validation")

    # Try Mode A first — merge existing per-model files
    merged = merge_existing_files()

    if len(merged) == 0:
        print("\nNo per-model files found. Running inference from scratch (Mode B)...")
        if EXTERNAL_COMBINED_PATH.exists():
            combined = pd.read_json(EXTERNAL_COMBINED_PATH, lines=True)
        else:
            combined = build_combined_dataset()
        merged = run_inference(combined)
        merged = merged[merged["predicted_label"] != "unknown"].copy()

    # Save merged
    merged.to_json(MERGED_PATH, orient="records", lines=True)
    print(f"\nSaved merged → {MERGED_PATH}")

    # Compute GRD
    ext_grd = compute_external_grd(merged)
    save_json(ext_grd, EXTERNAL_GRD_PATH)
    print(f"Saved → {EXTERNAL_GRD_PATH}")
    print("\n✅ Step 6 complete.")


if __name__ == "__main__":
    main()
