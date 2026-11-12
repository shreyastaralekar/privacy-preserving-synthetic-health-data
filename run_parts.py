#!/usr/bin/env python
"""Run selected modular project scripts in sequence."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PART_SCRIPTS = {
    2: "part02_preprocessing.py",
    3: "part03_eda.py",
    4: "part04_baseline_ml.py",
    5: "part05_ddpm_architecture.py",
    6: "part06_nonprivate_training.py",
    7: "part07_dp_sgd_training.py",
    8: "part08_privacy_accounting.py",
    9: "part09_synthetic_generation.py",
    10: "part10_statistical_fidelity.py",
    11: "part11_tstr_trtr_utility.py",
    12: "part12_privacy_audits.py",
    13: "part13_utility_privacy_tradeoff.py",
    14: "part14_final_report_builder.py",
}

def main():
    parser = argparse.ArgumentParser(description="Run modular Parts 2-14")
    parser.add_argument("--parts", nargs="+", type=int, required=True,
                        help="Parts to run in order, e.g. --parts 2 3 4")
    args = parser.parse_args()
    for part in args.parts:
        if part not in PART_SCRIPTS:
            parser.error(f"Part {part} is not in the modular runner. Use Parts 2-14; Part 1 is reference-only.")
    for part in args.parts:
        script = ROOT / "scripts" / PART_SCRIPTS[part]
        print(f"\n\n######## RUNNING PART {part} ########\n", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"Part {part} failed with exit code {result.returncode}. Later parts were not started.")
            raise SystemExit(result.returncode)
    print("\nRequested parts completed successfully.")

if __name__ == "__main__":
    main()
