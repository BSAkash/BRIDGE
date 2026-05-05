"""
config.py — Central configuration for all paths, constants, and model registry.
Edit this file to point to your data and add/remove models.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR     = ROOT / "figures"

# Input files — update these to your actual paths
PREDICTIONS_PATH         = DATA_DIR / "predictions_zeroshot.jsonl"
MERGED_PREDICTIONS_PATH  = DATA_DIR / "merged_predictions.jsonl"   # raw before fix
EXTERNAL_COMBINED_PATH   = DATA_DIR / "external_combined.jsonl"
EXTERNAL_PREDICTIONS_PATH= DATA_DIR / "external_predictions.jsonl"

# Output files
STRATIFIED_RESULTS_PATH  = RESULTS_DIR / "stratified_results.csv"
RANKINGS_ALL_PATH        = RESULTS_DIR / "rankings_rbb_all.csv"
RANKINGS_FORMAL_PATH     = RESULTS_DIR / "rankings_rbb_formal.csv"
RANKINGS_INFORMAL_PATH   = RESULTS_DIR / "rankings_rbb_informal.csv"
RANKING_FLIP_PATH        = RESULTS_DIR / "ranking_flip.csv"
GRD_RESULTS_PATH         = RESULTS_DIR / "grd_results.json"
RRS_PATH                 = RESULTS_DIR / "rrs_per_model_class.csv"
CD_PATH                  = RESULTS_DIR / "confusion_drift.csv"
CCD_PATH                 = RESULTS_DIR / "capability_drift.csv"
EIL_RESULTS_PATH         = RESULTS_DIR / "eil_results.json"
IRT_RESULTS_PATH         = RESULTS_DIR / "irt_results.csv"
CONSTRUCT_SHIFT_PATH     = RESULTS_DIR / "construct_shift.json"
SENSITIVITY_PATH         = RESULTS_DIR / "sensitivity_curves.csv"
SENSITIVITY_BP_PATH      = RESULTS_DIR / "sensitivity_breakpoints.json"
ABLATION_PATH            = RESULTS_DIR / "ablation_results.csv"
FAILURE_PATH             = RESULTS_DIR / "failure_taxonomy.csv"
FAILURE_REVIEW_PATH      = RESULTS_DIR / "failure_manual_review_sample.csv"
EXTERNAL_GRD_PATH        = RESULTS_DIR / "external_grd_results.json"

# Create dirs
for d in [DATA_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Labels ────────────────────────────────────────────────────────────────────
LABELS = ["harmful", "harmless", "antibias"]

LABEL_MAP = {
    "harmful bias":  "harmful",
    "no bias":       "harmless",
    "anti bias":     "antibias",
    "harmful":       "harmful",
    "harmless":      "harmless",
    "antibias":      "antibias",
    "anti-bias":     "antibias",
    "invalid":       "unknown",
}

# ── Register assignment ───────────────────────────────────────────────────────
REGISTER_MAP = {
    "Social Bias Frames":                   "informal",
    "Pile of Law":                          "formal",
    "On the Books: Jim Crow Laws":          "formal",
    "Mapping Inequality: Redlining Data":   "formal",
    "GovReport Summarization":              "formal",
    "Comparative Agendas Project (CAP)":    "formal",
    "LSC Eviction Laws Database":           "formal",
}

ERA_MAP = {
    "Social Bias Frames":                   "contemporary",
    "Pile of Law":                          "modern",
    "On the Books: Jim Crow Laws":          "pre-1965",
    "Mapping Inequality: Redlining Data":   "pre-1965",
    "GovReport Summarization":              "modern",
    "Comparative Agendas Project (CAP)":    "modern",
    "LSC Eviction Laws Database":           "modern",
}

# ── Models ────────────────────────────────────────────────────────────────────
# Left key = model name in your predictions file
# Right value = OpenRouter model string
MODEL_REGISTRY = {
    "GPT-5.4":                      "openai/gpt-4o-mini",
    "Claude Sonnet 4.6":            "anthropic/claude-sonnet-4-5",
    "Gemini 3.1 Pro Preview":       "google/gemini-pro-1.5",
    "Llama 3.1 70B-Inst 4-bit":     "meta-llama/llama-3.1-70b-instruct",
    "Llama 3.1 8B-Inst":            "meta-llama/llama-3.1-8b-instruct",
    "Qwen 2.5 72B Instruct 4-bit":  "qwen/qwen-2.5-72b-instruct",
    "Mistral 7B v0.3 Instruct":     "mistralai/mistral-7b-instruct-v0.3",
    "Gemma 2 9B Instruct 4-bit":    "google/gemma-2-9b-it",
}

# ── External validation config ────────────────────────────────────────────────
# Prompt used for external validation inference
EXTERNAL_PROMPT = """You are a bias detection system. Classify the following sentence into exactly one category:
- "harmful": perpetuates discriminatory bias or hate speech
- "harmless": neutral, no bias present

Respond with only one word: harmful or harmless

Sentence: {text}"""

# MeasuringHateSpeech score thresholds
MHS_HARMFUL_THRESHOLD  =  1.0   # mean_score > 1.0  → harmful
MHS_HARMLESS_THRESHOLD = -0.5   # mean_score < -0.5 → harmless
# scores in between are dropped as ambiguous

# ── Sensitivity curve config ─────────────────────────────────────────────────
SENSITIVITY_EVAL_SIZE = 2000
SENSITIVITY_N_SEEDS   = 5
SENSITIVITY_RATIOS    = [i / 10 for i in range(11)]

# ── Failure analysis config ───────────────────────────────────────────────────
N_HARD_SENTENCES      = 500
N_MANUAL_REVIEW       = 200
