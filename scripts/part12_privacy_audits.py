#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part12_DP_Medical_Privacy_Audits.ipynb.

Run from the project root with: python scripts/part12_privacy_audits.py
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
print("Part 12 – Part12_DP_Medical_Privacy_Audits")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 12 – Privacy-Risk Diagnostics: Exact Matches, DCR and Membership Inference
#
# This notebook performs three complementary empirical checks on the synthetic data generated in Part 9:
#
# 1. **Exact-match audit:** count synthetic records that exactly match a training respondent across the selected categorical fields.
# 2. **Distance to Closest Record (DCR):** measure normalized Hamming distance from each synthetic/query row to its nearest reference row.
# 3. **Distance-based membership-inference diagnostic:** test whether records sampled from the training split can be distinguished from records sampled from the held-out test split using their distance to the released synthetic sample.
#
# These are empirical risk diagnostics, not a formal proof of privacy. The formal differential privacy guarantee, when applicable, comes from the validated DP-SGD mechanism and its accountant (Parts 7 and 8). BRFSS is public survey microdata, so results here should not be represented as protection of confidential hospital records.

# ## 1. Install and import dependencies

# [Notebook install command moved to requirements.txt]

from pathlib import Path
import os, json, math, random, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = configure_project(__file__)
STORAGE_MODE = "Local IDE project folder"

DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
MODEL_DIR = PROJECT_ROOT / 'models'
METRICS_DIR = PROJECT_ROOT / 'metrics'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
SYNTHETIC_DIR = ARTIFACT_DIR / 'synthetic_data'
for folder in [PROJECT_ROOT, DATA_DIR, PLOTS_DIR, CHECKPOINT_DIR, MODEL_DIR, METRICS_DIR, ARTIFACT_DIR, SYNTHETIC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = DATA_DIR / 'ddpm_config.json'
SCHEMA_PATH = DATA_DIR / 'schema_manifest.json'
if not CONFIG_PATH.exists() or not SCHEMA_PATH.exists():
    raise FileNotFoundError('Missing Part 5 artifacts. Run Parts 1–5 first and confirm data/ddpm_config.json and data/schema_manifest.json exist.')
config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
FEATURE_COLS = list(schema['feature_columns'])
TARGET_COL = str(schema['target_column'])
ALL_COLS = list(schema['all_columns'])
DOMAINS = {k: list(v) for k, v in schema['domains'].items()}
FEATURE_SLICES = {k: tuple(v) for k, v in schema['feature_slices'].items()}
INPUT_DIM = int(schema['input_dim'])
DEVICE = None
print('Storage:', STORAGE_MODE)
print('Project:', PROJECT_ROOT)
print('Feature fields:', len(FEATURE_COLS), '| target:', TARGET_COL, '| encoded dimension:', INPUT_DIM)

def canonical_code(value, column):
    """Convert numeric-looking CSV values to the fixed schema's string code."""
    missing_token = schema.get('missing_token', '__MISSING__')
    other_token = schema.get('other_token', '__OTHER__')
    if pd.isna(value):
        return missing_token
    s = str(value).strip()
    if s in DOMAINS[column]:
        return s
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return missing_token
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            s = str(int(f))
    except Exception:
        pass
    return s if s in DOMAINS[column] else other_token


def canonicalize_frame(frame):
    out = frame.copy()
    out.columns = [str(c).upper() for c in out.columns]
    missing = [c for c in ALL_COLS if c not in out.columns]
    if missing:
        raise KeyError(f'Required columns absent: {missing}')
    out = out[ALL_COLS]
    for col in ALL_COLS:
        out[col] = out[col].map(lambda v: canonical_code(v, col))
    return out

from sklearn.metrics import roc_auc_score, roc_curve

# ## 2. Load the real splits, synthetic sample and selected model metadata

MODEL_LABEL = 'dp_eps_3p0'  # Must match the Part 9 output label.
SYNTHETIC_PATH = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}.csv'
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
META_PATH = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}_metadata.json'
for p in [SYNTHETIC_PATH, TRAIN_PATH, TEST_PATH]:
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Complete Parts 2 and 9 first.')
synthetic_df = canonicalize_frame(pd.read_csv(SYNTHETIC_PATH, dtype=str, keep_default_na=False))
train_df = canonicalize_frame(pd.read_csv(TRAIN_PATH))
test_df = canonicalize_frame(pd.read_csv(TEST_PATH))
run_meta = json.loads(META_PATH.read_text(encoding='utf-8')) if META_PATH.exists() else {}
if run_meta and run_meta.get('data_fingerprint') != config.get('data_fingerprint'):
    raise ValueError('Synthetic data metadata fingerprint does not match the current Part 5 configuration.')
print('Synthetic:', synthetic_df.shape, '| train:', train_df.shape, '| test:', test_df.shape)
print('Synthetic model:', MODEL_LABEL)

# ## 3. Encode categorical records and compute nearest Hamming distances
#
# Each field is mapped to its fixed schema index. The normalized Hamming distance is the fraction of fields that differ between two rows, ranging from 0 (identical) to 1 (different in every field). This simple mixed categorical distance gives each field equal weight; it is not a clinical severity measure.

def canonical_code(value, column):
    """Convert numeric-looking CSV values to the fixed schema's string code."""
    missing_token = schema.get('missing_token', '__MISSING__')
    other_token = schema.get('other_token', '__OTHER__')
    if pd.isna(value):
        return missing_token
    s = str(value).strip()
    if s in DOMAINS[column]:
        return s
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return missing_token
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            s = str(int(f))
    except Exception:
        pass
    return s if s in DOMAINS[column] else other_token


def canonicalize_frame(frame):
    out = frame.copy()
    out.columns = [str(c).upper() for c in out.columns]
    missing = [c for c in ALL_COLS if c not in out.columns]
    if missing:
        raise KeyError(f'Required columns absent: {missing}')
    out = out[ALL_COLS]
    for col in ALL_COLS:
        out[col] = out[col].map(lambda v: canonical_code(v, col))
    return out

def frame_to_codes(frame):
    arrays = []
    for col in ALL_COLS:
        lookup = {value: idx for idx, value in enumerate(DOMAINS[col])}
        arrays.append(frame[col].map(lookup).to_numpy(dtype=np.int16))
    return np.column_stack(arrays)

syn_codes = frame_to_codes(synthetic_df)
train_codes = frame_to_codes(train_df)
test_codes = frame_to_codes(test_df)


def nearest_hamming_distance(query_codes, reference_codes, chunk_size=64):
    """Return each query's minimum normalized Hamming distance to the reference set."""
    if len(reference_codes) == 0:
        raise ValueError('Reference set cannot be empty.')
    mins = np.empty(len(query_codes), dtype=np.float32)
    for start in range(0, len(query_codes), chunk_size):
        q = query_codes[start:start+chunk_size]
        # 3-D boolean comparison is kept chunked to limit Colab RAM use.
        distances = (q[:, None, :] != reference_codes[None, :, :]).mean(axis=2)
        mins[start:start+len(q)] = distances.min(axis=1)
    return mins

print('Encoded code matrices:', syn_codes.shape, train_codes.shape, test_codes.shape)

# ## 4. Exact-match and DCR audit

# Exact matching against all training rows using tuples of category indices.
train_tuple_set = set(map(tuple, train_codes.tolist()))
syn_tuples = list(map(tuple, syn_codes.tolist()))
exact_match_flags = np.fromiter((row in train_tuple_set for row in syn_tuples), dtype=bool, count=len(syn_tuples))
exact_match_count = int(exact_match_flags.sum())
synthetic_unique_rows = int(len(set(syn_tuples)))

# Subsample the training reference for an efficient nearest-neighbour estimate.
MAX_TRAIN_REFERENCE = 10_000
rng = np.random.default_rng(SEED)
if len(train_codes) > MAX_TRAIN_REFERENCE:
    ref_idx = rng.choice(len(train_codes), MAX_TRAIN_REFERENCE, replace=False)
    train_reference = train_codes[ref_idx]
else:
    train_reference = train_codes

MAX_SYN_DCR_QUERIES = 5_000
syn_query = syn_codes if len(syn_codes) <= MAX_SYN_DCR_QUERIES else syn_codes[rng.choice(len(syn_codes), MAX_SYN_DCR_QUERIES, replace=False)]
syn_to_train_dcr = nearest_hamming_distance(syn_query, train_reference, chunk_size=64)

# Also calculate reference split distances to the synthetic release for a membership-risk diagnostic.
MAX_MEMBERSHIP_QUERIES = 1_000
train_query = train_codes if len(train_codes) <= MAX_MEMBERSHIP_QUERIES else train_codes[rng.choice(len(train_codes), MAX_MEMBERSHIP_QUERIES, replace=False)]
test_query = test_codes if len(test_codes) <= MAX_MEMBERSHIP_QUERIES else test_codes[rng.choice(len(test_codes), MAX_MEMBERSHIP_QUERIES, replace=False)]
MAX_SYN_REFERENCE = 10_000
syn_reference = syn_codes if len(syn_codes) <= MAX_SYN_REFERENCE else syn_codes[rng.choice(len(syn_codes), MAX_SYN_REFERENCE, replace=False)]
train_to_syn_dcr = nearest_hamming_distance(train_query, syn_reference, chunk_size=32)
test_to_syn_dcr = nearest_hamming_distance(test_query, syn_reference, chunk_size=32)

mia_labels = np.concatenate([np.ones(len(train_to_syn_dcr), dtype=int), np.zeros(len(test_to_syn_dcr), dtype=int)])
mia_scores = -np.concatenate([train_to_syn_dcr, test_to_syn_dcr])  # closer to synthetic => higher membership score
mia_auc_raw = float(roc_auc_score(mia_labels, mia_scores)) if len(np.unique(mia_labels)) == 2 else float('nan')
mia_auc = float(max(mia_auc_raw, 1.0 - mia_auc_raw)) if np.isfinite(mia_auc_raw) else float('nan')

summary = {
    'model_label': MODEL_LABEL,
    'synthetic_rows': int(len(syn_codes)),
    'training_rows': int(len(train_codes)),
    'test_rows': int(len(test_codes)),
    'exact_synthetic_matches_to_training': exact_match_count,
    'exact_match_rate': float(exact_match_flags.mean()) if len(exact_match_flags) else 0.0,
    'unique_synthetic_rows': synthetic_unique_rows,
    'synthetic_duplicate_row_rate': float(1.0 - synthetic_unique_rows / max(1, len(syn_codes))),
    'synthetic_to_training_dcr_mean': float(np.mean(syn_to_train_dcr)),
    'synthetic_to_training_dcr_median': float(np.median(syn_to_train_dcr)),
    'synthetic_to_training_dcr_p05': float(np.quantile(syn_to_train_dcr, 0.05)),
    'training_to_synthetic_dcr_mean': float(np.mean(train_to_syn_dcr)),
    'test_to_synthetic_dcr_mean': float(np.mean(test_to_syn_dcr)),
    'distance_based_membership_attack_auc': mia_auc,
    'distance_based_membership_attack_auc_raw_train_as_positive': mia_auc_raw,
    'distance_metric': 'normalized Hamming distance across 24 categorical fields',
    'membership_attack': 'train vs held-out test discrimination using negative nearest distance to synthetic sample',
    'reference_sampling': {'training_reference_n': int(len(train_reference)), 'synthetic_reference_n': int(len(syn_reference)),
                           'synthetic_dcr_query_n': int(len(syn_query)), 'train_attack_query_n': int(len(train_query)),
                           'test_attack_query_n': int(len(test_query))},
    'interpretation': 'Empirical diagnostics only; not a formal DP proof. The reported attack AUC is max(raw AUC, 1-raw AUC), allowing an adversary to reverse the score direction. A value near 0.5 means this particular distance attack did not distinguish the sampled groups well, not that privacy risk is zero.'
}
print(json.dumps(summary, indent=2))

# ## 5. Confirm the formal DP status separately from empirical audits
#
# A DP guarantee is only reported for a DP model when the matching Part 7 run metadata and Part 8 accountant verification are present. A low exact-match rate or a membership attack AUC near 0.5 does not replace this check.

privacy_status = {'model_label': MODEL_LABEL, 'formal_dp_verified': False, 'reason': 'No matching verified DP accounting record was found.'}
if MODEL_LABEL.startswith('dp_eps_'):
    # Part 7 uses the same epsilon tag naming as MODEL_LABEL.
    meta_path = METRICS_DIR / f'dp_run_eps_{MODEL_LABEL.replace("dp_eps_", "")}.json'
    verification_path = METRICS_DIR / 'privacy_accounting_verification.csv'
    if meta_path.exists() and verification_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('data_fingerprint') != config.get('data_fingerprint'):
            raise ValueError('Part 7 DP metadata does not match the current Part 5 data fingerprint.')
        verification = pd.read_csv(verification_path)
        match = verification[np.isclose(verification['target_epsilon'].astype(float), float(meta['target_epsilon']))]
        if len(match):
            row = match.iloc[0]
            privacy_status = {
                'model_label': MODEL_LABEL,
                'formal_dp_verified': bool(row['verification'] == 'PASS' and row['target_budget_status'] == 'WITHIN_TARGET'),
                'target_epsilon': float(meta['target_epsilon']),
                'achieved_epsilon': float(row['recomputed_epsilon']),
                'delta': float(meta['delta']),
                'accountant_verification': str(row['verification']),
                'target_budget_status': str(row['target_budget_status']),
                'secure_mode': bool(meta.get('secure_mode', False)),
                'privacy_unit': meta.get('privacy_unit', 'one BRFSS respondent row'),
            }
print(json.dumps(privacy_status, indent=2))

# ## 6. Save audit outputs and visualizations

SUMMARY_PATH = METRICS_DIR / f'part12_privacy_audit_summary_{MODEL_LABEL}.json'
DISTANCE_PATH = METRICS_DIR / f'part12_dcr_samples_{MODEL_LABEL}.csv'
ATTACK_PATH = METRICS_DIR / f'part12_membership_attack_samples_{MODEL_LABEL}.csv'
summary['formal_dp_accounting_status'] = privacy_status
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
pd.DataFrame({'synthetic_to_training_nearest_hamming': syn_to_train_dcr}).to_csv(DISTANCE_PATH, index=False)
pd.DataFrame({
    'group': ['train'] * len(train_to_syn_dcr) + ['held_out_test'] * len(test_to_syn_dcr),
    'nearest_synthetic_hamming_distance': np.concatenate([train_to_syn_dcr, test_to_syn_dcr]),
}).to_csv(ATTACK_PATH, index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(syn_to_train_dcr, bins=25, alpha=0.75, label='Synthetic → train nearest distance')
ax.set_xlabel('Normalized Hamming distance'); ax.set_ylabel('Count')
ax.set_title('Distance to closest training respondent (sampled estimate)')
ax.grid(axis='y', alpha=0.2); ax.legend()
fig.tight_layout()
DCR_PLOT = PLOTS_DIR / f'part12_dcr_distribution_{MODEL_LABEL}.png'
fig.savefig(DCR_PLOT, dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(train_to_syn_dcr, bins=25, alpha=0.6, density=True, label='Training queries')
ax.hist(test_to_syn_dcr, bins=25, alpha=0.6, density=True, label='Held-out test queries')
ax.set_xlabel('Nearest distance to synthetic sample'); ax.set_ylabel('Density')
ax.set_title(f'Distance-based membership diagnostic (AUC={mia_auc:.3f})')
ax.grid(axis='y', alpha=0.2); ax.legend()
fig.tight_layout()
MIA_PLOT = PLOTS_DIR / f'part12_membership_distance_{MODEL_LABEL}.png'
fig.savefig(MIA_PLOT, dpi=180, bbox_inches='tight')
plt.show()

# Human-readable audit note suitable for attaching to the final project report.
report = f"""# Privacy-Risk Diagnostics – {MODEL_LABEL}

## Scope
This is an empirical audit of generated categorical survey records. It is not a formal privacy proof.

## Results
- Synthetic rows: {summary['synthetic_rows']}
- Exact matches to training rows: {summary['exact_synthetic_matches_to_training']} ({summary['exact_match_rate']:.6%})
- Unique synthetic rows: {summary['unique_synthetic_rows']}
- Synthetic duplicate-row rate: {summary['synthetic_duplicate_row_rate']:.6%}
- Synthetic-to-training DCR median: {summary['synthetic_to_training_dcr_median']:.6f}
- Synthetic-to-training DCR 5th percentile: {summary['synthetic_to_training_dcr_p05']:.6f}
- Distance-based membership attack AUC: {summary['distance_based_membership_attack_auc']:.6f}

## Formal DP accounting status
```json
{json.dumps(privacy_status, indent=2)}
```

## Interpretation
A low exact-match rate or a membership attack AUC near 0.5 is not proof that disclosure risk is absent. The distance attack is limited to the chosen features, distance metric, sample sizes and synthetic release. A formal DP statement must use the validated accountant output and the actual privacy unit. The project source is public BRFSS survey data, not confidential hospital records.
"""
REPORT_PATH = METRICS_DIR / f'part12_privacy_audit_report_{MODEL_LABEL}.md'
REPORT_PATH.write_text(report, encoding='utf-8')
print('Saved audit summary:', SUMMARY_PATH)
print('Saved report:', REPORT_PATH)

# ## 7. Required interpretation checklist
#
# Before writing a conclusion, confirm:
#
# - [ ] The synthetic model label matches the model used in Parts 9–11.
# - [ ] Exact-match search was made against the complete saved training split.
# - [ ] DCR is described as normalized Hamming distance across the selected categorical fields.
# - [ ] The training and test attack query sample sizes are disclosed.
# - [ ] AUC near 0.5 is not described as proof of zero privacy risk.
# - [ ] Any formal DP claim is taken from Part 8, not inferred from these attack results.
# - [ ] If multiple DP models or releases are shared, composition is considered.
# - [ ] The public BRFSS survey-data limitation is clearly stated.
#
# ## References
#
# 1. Shokri, R., et al. (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE Symposium on Security and Privacy*.
# 2. Carlini, N., et al. (2022). Membership Inference Attacks From First Principles. *IEEE Symposium on Security and Privacy*.
# 3. Opacus documentation. https://opacus.ai/
# 4. CDC. 2025 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2025.html
