"""
RBB Paper — Full Pipeline Runner
=================================
Run this script to execute the entire experimental pipeline in order.

Usage:
    python run_all.py --step all          # run everything
    python run_all.py --step prepare      # step 0: fix + merge predictions
    python run_all.py --step results      # step 1: stratified results + rankings
    python run_all.py --step grd          # step 2: GRD + DC + RPF + CD + CCD
    python run_all.py --step eil          # step 3: EIL + IRT (supplementary)
    python run_all.py --step sensitivity  # step 4: sensitivity curves
    python run_all.py --step ablations    # step 5: ablations + failure analysis
    python run_all.py --step external     # step 6: external validation (BABE + MHS)
    python run_all.py --step figures      # step 7: all paper figures
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_step(step: str):
    if step in ("all", "prepare"):
        print("\n" + "="*60)
        print("STEP 0: Prepare Predictions")
        print("="*60)
        from src.s0_prepare import main as s0
        s0()

    if step in ("all", "results"):
        print("\n" + "="*60)
        print("STEP 1: Stratified Results + Rankings")
        print("="*60)
        from src.s1_results import main as s1
        s1()

    if step in ("all", "grd"):
        print("\n" + "="*60)
        print("STEP 2: GRD + DC + RPF + CD + CCD")
        print("="*60)
        from src.s2_grd import main as s2
        s2()

    if step in ("all", "eil"):
        print("\n" + "="*60)
        print("STEP 3: EIL + IRT (Supplementary)")
        print("="*60)
        from src.s3_eil_irt import main as s3
        s3()

    if step in ("all", "sensitivity"):
        print("\n" + "="*60)
        print("STEP 4: Sensitivity Curves")
        print("="*60)
        from src.s4_sensitivity import main as s4
        s4()

    if step in ("all", "ablations"):
        print("\n" + "="*60)
        print("STEP 5: Ablations + Failure Analysis")
        print("="*60)
        from src.s5_ablations import main as s5
        s5()

    if step in ("all", "external"):
        print("\n" + "="*60)
        print("STEP 6: External Validation")
        print("="*60)
        from src.s6_external import main as s6
        s6()

    if step in ("all", "figures"):
        print("\n" + "="*60)
        print("STEP 7: Generate All Figures")
        print("="*60)
        from src.s7_figures import main as s7
        s7()

    print("\n✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all",
        choices=["all","prepare","results","grd","eil","sensitivity","ablations","external","figures"])
    args = parser.parse_args()
    run_step(args.step)
