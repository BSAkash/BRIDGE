# RBB Paper — Codebase v2

**"Evaluation Equalized Odds: Bias Benchmarks Exhibit a Systematic Register Polarity Flip Across All Frontier LLMs"**
NeurIPS 2026 Datasets & Benchmarks Track

## What Changed in v2

- `s4_sensitivity.py` — Fixed matplotlib legend bug in plot_sensitivity()
- `s5_ablations.py` — Handles missing `text` column; skips length/readability ablations gracefully; source ablation always runs
- `s6_external.py` — Auto-detects and merges per-model prediction files; model name normalization built in
- `s7_figures.py` — Updated with actual result keys; Figure 7 shows GRD + DC side by side

## Project Structure

```
rbb_paper/
├── run_all.py              # Master runner
├── requirements.txt
├── README.md
├── data/                   # Put your files here
│   ├── predictions_zeroshot.jsonl        ← main predictions (normalized)
│   ├── merged_predictions.jsonl          ← raw before normalization
│   ├── external_combined.jsonl           ← BABE + MHS combined
│   ├── external_predictions_merged.jsonl ← merged external predictions
│   └── [per-model external files]
├── results/                # All outputs
└── src/
    ├── config.py           ← EDIT THIS FIRST
    ├── utils.py
    ├── s0_prepare.py       ← fix labels + normalize
    ├── s1_results.py       ← stratified results + rankings
    ├── s2_grd.py           ← GRD + DC + RPF + CD + CCD  ← PRIMARY
    ├── s3_eil_irt.py       ← EIL + IRT (supplementary)
    ├── s4_sensitivity.py   ← sensitivity curves (fixed)
    ├── s5_ablations.py     ← ablations + failure (fixed)
    ├── s6_external.py      ← external validation (updated)
    └── s7_figures.py       ← all figures (updated)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running on Colab

### 1. Upload files to /content/
- `predictions_zeroshot.jsonl` (your main predictions)
- All `src/*.py` files

### 2. Install dependencies
```python
!pip install -q scikit-learn scipy matplotlib seaborn textstat datasets openai
!python -m spacy download en_core_web_sm -q
```

### 3. Set Colab paths (run before anything else)
```python
import sys, types
sys.path.insert(0, '/content')
config = types.ModuleType('config')
# ... (see full config block in team summary doc)
sys.modules['config'] = config
```

### 4. Run steps
```python
import s0_prepare; s0_prepare.main()   # fix labels
import s1_results; s1_results.main()   # rankings
import s2_grd;     s2_grd.main()       # GRD ← most important
import s4_sensitivity; s4_sensitivity.main()
import s5_ablations;   s5_ablations.main()  # needs s2 first
import s6_external;    s6_external.main()   # needs per-model files
import s7_figures;     s7_figures.main()    # needs everything
```

## Key Results

| Metric | Value |
|---|---|
| GRD_global (register) | 0.2710 |
| DC(harmful) | 1.0000 |
| DC(harmless) | 1.0000 |
| RPF | 1 (Polarity Flip) |
| CD(harmful→harmless) | −0.307 |
| GRD_global (era) | 0.2580 |
| GRD_global (external) | 0.5036 |
| DC(harmful) external | 1.0000 |
| Sensitivity breakpoint | 10% formal |
| Hardest sentences formal | 87.6% |

## External Validation

Place per-model files in `/content/` named like:
```
*external_predictions*.jsonl
```

s6_external.py will auto-detect, merge, normalize model names, and compute GRD.
