#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part8_DP_Medical_Privacy_Accounting.ipynb.

Run from the project root with: python scripts/part08_privacy_accounting.py
This script uses local disk storage; it does not mount Google Drive.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # save plots without GUI pop-up requirements
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import configure_project, display

PROJECT_ROOT = configure_project(__file__)
STORAGE_MODE = "Local IDE project folder"
print("\n" + "=" * 76)
print("Part 08 – Part8_DP_Medical_Privacy_Accounting")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 8 – Privacy Accounting, Verification and Composition Analysis
#
# **Goal:** Recompute each completed DP model's ε from the saved Opacus accountant state, verify the recorded result, and produce a report-ready privacy table and plot.
#
# **Inputs:** `checkpoints/accountant_dp_eps_*.pt` and `metrics/dp_run_eps_*.json` from Part 7.  
# **Outputs:** verified accountant CSV, composition summary, verification plot and Markdown report.
#
# This notebook does not retrain models. It checks that the saved accountant history can be loaded and produces the same ε at the recorded δ.

# ## 1. Install Opacus and mount persistent storage

# [Notebook install command moved to requirements.txt]

from pathlib import Path
import os, json, math, random, hashlib, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One persistent Google Drive folder is shared by every project part.
PROJECT_ROOT = configure_project(__file__)
STORAGE_MODE = "Local IDE project folder"

DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
MODEL_DIR = PROJECT_ROOT / 'models'
METRICS_DIR = PROJECT_ROOT / 'metrics'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
for folder in [PROJECT_ROOT, DATA_DIR, PLOTS_DIR, CHECKPOINT_DIR, MODEL_DIR, METRICS_DIR, ARTIFACT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print('Storage:', STORAGE_MODE)
print('Project:', PROJECT_ROOT)
print('Device:', DEVICE)

# ## 2. Load the accountant records and explain the privacy parameters

from opacus import PrivacyEngine
import opacus

meta_files = sorted(METRICS_DIR.glob('dp_run_eps_*.json'))
if not meta_files:
    raise FileNotFoundError('No completed DP run metadata found. Run Part 7 first.')

print('Opacus:', opacus.__version__)
print('Completed run metadata files:', len(meta_files))
print('Privacy unit: one BRFSS respondent row')
print('The delta values below were selected in Part 7 as min(1e-5, 1/(10*N_train)).')

# ## 3. Reconstruct each accountant and verify achieved epsilon
#
# Opacus accountants expose a serialisable state dictionary. Restoring this state into an accountant with the same mechanism allows ε to be recomputed at the original δ.

verified_rows = []
for meta_path in meta_files:
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    tag = str(float(meta['target_epsilon'])).replace('.', 'p')
    accountant_path = CHECKPOINT_DIR / f'accountant_dp_eps_{tag}.pt'
    model_path = CHECKPOINT_DIR / f'ddpm_dp_eps_{tag}.pt'
    if not accountant_path.exists() or not model_path.exists():
        warnings.warn(f"Skipping {meta['model']}: model or accountant checkpoint is missing.")
        continue

    state = torch.load(accountant_path, map_location='cpu', weights_only=False)
    mechanism = str(meta['accountant'])
    engine = PrivacyEngine(accountant=mechanism, secure_mode=False)
    engine.accountant.load_state_dict(state)
    recomputed_epsilon = float(engine.get_epsilon(delta=float(meta['delta'])))
    recorded_epsilon = float(meta['achieved_epsilon'])
    abs_diff = abs(recomputed_epsilon - recorded_epsilon)
    tolerance = max(1e-6, 1e-4 * max(1.0, abs(recorded_epsilon)))
    state_status = 'PASS' if abs_diff <= tolerance else 'FAIL'
    target_status = 'WITHIN_TARGET' if recomputed_epsilon <= float(meta['target_epsilon']) + tolerance else 'EXCEEDS_TARGET'
    history = state.get('history', [])
    accountant_steps = int(sum(int(item[-1]) for item in history))

    verified_rows.append({
        'model': meta['model'],
        'target_epsilon': float(meta['target_epsilon']),
        'recorded_epsilon': recorded_epsilon,
        'recomputed_epsilon': recomputed_epsilon,
        'absolute_difference': abs_diff,
        'delta': float(meta['delta']),
        'delta_less_than_1_over_N': bool(float(meta['delta']) < 1.0 / int(meta['respondent_train_n'])),
        'noise_multiplier': float(meta['noise_multiplier']),
        'max_grad_norm': float(meta['max_grad_norm']),
        'respondent_train_n': int(meta['respondent_train_n']),
        'accountant': mechanism,
        'accountant_steps_recomputed': accountant_steps,
        'verification': state_status,
        'target_budget_status': target_status,
        'secure_mode': bool(meta.get('secure_mode', False)),
    })

if not verified_rows:
    raise RuntimeError('No complete model/accountant pairs were available to verify.')
verification_df = pd.DataFrame(verified_rows).sort_values('target_epsilon').reset_index(drop=True)
display(verification_df)
verification_path = METRICS_DIR / 'privacy_accounting_verification.csv'
verification_df.to_csv(verification_path, index=False)
print('Saved:', verification_path)

# ## 4. Basic composition when multiple DP models are released
#
# Each ε target was trained as a separate mechanism on the same respondent-level training set. If all of the resulting checkpoints or generated datasets are released, basic sequential composition gives an upper bound by summing the individual ε values and summing their δ values (capped at 1). This is conservative; tighter composition requires an explicitly selected theorem and compatible assumptions. If only one model is released, report that model's own measured budget and document the release/selection process.

individual_eps = verification_df['recomputed_epsilon'].astype(float).tolist()
individual_deltas = verification_df['delta'].astype(float).tolist()
num_releases = len(verification_df)
composition = {
    'number_of_completed_dp_models': int(num_releases),
    'basic_composed_epsilon_if_all_models_released': float(sum(individual_eps)),
    'basic_composed_delta_if_all_models_released': float(min(1.0, sum(individual_deltas))),
    'individual_epsilon_values': individual_eps,
    'individual_delta_values': individual_deltas,
    'composition_rule': 'basic sequential composition: epsilon_total <= sum(epsilon_i), delta_total <= sum(delta_i)',
    'interpretation': 'Use individual values when only one DP mechanism/model is released and the release process itself is accounted for. Use composition when multiple model outputs are made available.'
}
composition_path = METRICS_DIR / 'dp_composition_summary.json'
composition_path.write_text(json.dumps(composition, indent=2), encoding='utf-8')
print(json.dumps(composition, indent=2))

# ## 5. Create report-ready privacy figures and a Markdown summary

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(verification_df))
width = 0.36
ax.bar(x - width/2, verification_df['target_epsilon'], width, label='Target ε')
ax.bar(x + width/2, verification_df['recomputed_epsilon'], width, label='Accountant ε')
ax.set_xticks(x, verification_df['model'])
ax.set_ylabel('Epsilon (ε)')
ax.set_title('Target privacy budget versus verified accountant result')
ax.grid(axis='y', alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'part8_verified_privacy_budgets.png', dpi=180, bbox_inches='tight')
plt.show()

rows_md = []
for _, row in verification_df.iterrows():
    rows_md.append(
        f"| {row['model']} | {row['target_epsilon']:.4f} | {row['recorded_epsilon']:.6f} | "
        f"{row['recomputed_epsilon']:.6f} | {row['delta']:.2e} | {row['noise_multiplier']:.5f} | {row['verification']} | {row['target_budget_status']} |"
    )
report_md = '''# Privacy Accounting Verification – DP Diffusion BRFSS Project

## Scope
This report verifies saved Opacus accountant states for separate respondent-level DP-SGD training runs. The protected unit is one respondent row in the selected 2025 BRFSS training split.

## Verified runs

| Model | Target ε | Recorded ε | Recomputed ε | δ | Noise multiplier | Accountant check | Target check |
|---|---:|---:|---:|---:|---:|---|---|
''' + '\n'.join(rows_md) + f'''\n
## Composition if every model is released

- Number of completed DP models: {num_releases}
- Basic composed epsilon upper bound: {composition['basic_composed_epsilon_if_all_models_released']:.6f}
- Basic composed delta upper bound: {composition['basic_composed_delta_if_all_models_released']:.2e}

## Interpretation and limitations

1. The accountant result is the achieved privacy budget for the recorded training mechanism; it is not inferred from the target label.
2. The non-private model has no formal DP guarantee and is not included in this table.
3. The guarantee relies on respondent-level adjacency, the configured Poisson sampling, per-example clipping, calibrated Gaussian noise and correct use of the Opacus-wrapped optimizer/data loader.
4. `secure_mode=False` was used for Colab compatibility. Do not describe this run as using cryptographically hardened randomness.
5. BRFSS is publicly accessible, self-reported survey microdata, not hospital EHR. This project demonstrates DP methodology; it does not make the public source file confidential.
6. Releasing multiple separately trained models trained on the same records incurs composition. Report the combined bound when all outputs are released.

## Files
- `metrics/privacy_accounting_verification.csv`
- `metrics/dp_composition_summary.json`
- `plots/part8_verified_privacy_budgets.png`
'''
report_path = METRICS_DIR / 'privacy_accounting_report.md'
report_path.write_text(report_md, encoding='utf-8')
print('Saved report:', report_path)
print('Saved plot:', PLOTS_DIR / 'part8_verified_privacy_budgets.png')

# ## 6. Privacy conclusion checklist
#
# Before including a privacy claim in the final report, confirm:
#
# - [ ] Every claimed DP model has a matching checkpoint, accountant state and run metadata.
# - [ ] The `verification` column says `PASS` for the claimed model.
# - [ ] The privacy unit is stated as one respondent row.
# - [ ] The δ value and actual achieved ε are reported.
# - [ ] The report distinguishes a single released model from the composition of multiple released models.
# - [ ] `secure_mode=False` is disclosed.
# - [ ] The project is described as BRFSS survey-data research, not hospital EHR research.

# ## References
#
# 1. Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*. ACM CCS. https://arxiv.org/abs/1607.00133
# 2. Opacus documentation, PrivacyEngine API. https://opacus.ai/api/privacy_engine.html
# 3. Opacus documentation, accountant interface and state restoration. https://opacus.ai/api/accounting/iaccountant.html
# 4. Centers for Disease Control and Prevention. 2025 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2025.html
