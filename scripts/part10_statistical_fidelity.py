#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part10_DP_Medical_Statistical_Fidelity.ipynb.

Run from the project root with: python scripts/part10_statistical_fidelity.py
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
print("Part 10 – Part10_DP_Medical_Statistical_Fidelity")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 10 – Statistical Fidelity Evaluation of Synthetic BRFSS Data
#
# **Purpose:** compare generated synthetic records with the untouched real test split from Part 2. This notebook computes marginal distribution similarity and pairwise categorical association similarity, saves machine-readable tables, and creates report-ready plots.
#
# **Why use the test split?** The generator was trained on the training respondents. Comparing generated data with the held-out test respondents gives a less training-biased view of whether the synthetic sample resembles the broader source population. The test data must not be used to fit the generator.
#
# **Metrics:** total variation distance (TVD), Jensen–Shannon divergence (JSD), and mean absolute difference between bias-corrected Cramér's V pairwise association matrices. These are descriptive dataset-level metrics; they are not privacy guarantees or evidence of clinical validity.

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

from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

# ## 2. Select and load the Part 9 synthetic output

MODEL_LABEL = 'dp_eps_3p0'  # Change to the model label used in Part 9, e.g. non_private or dp_eps_1p0.
SYNTHETIC_PATH = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}.csv'
REAL_TEST_PATH = DATA_DIR / 'test.csv'
if not SYNTHETIC_PATH.exists():
    available = sorted(p.name for p in SYNTHETIC_DIR.glob('synthetic_*.csv'))
    raise FileNotFoundError(f'Run Part 9 first. Available synthetic CSV files: {available}')
if not REAL_TEST_PATH.exists():
    raise FileNotFoundError('Missing data/test.csv. Run Part 2 first.')
synthetic_raw = pd.read_csv(SYNTHETIC_PATH, dtype=str, keep_default_na=False)
real_raw = pd.read_csv(REAL_TEST_PATH)
synthetic_df = canonicalize_frame(synthetic_raw)
synthetic_meta_path = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}_metadata.json'
if synthetic_meta_path.exists():
    synthetic_meta = json.loads(synthetic_meta_path.read_text(encoding='utf-8'))
    if synthetic_meta.get('data_fingerprint') != config.get('data_fingerprint'):
        raise ValueError('Synthetic data metadata fingerprint does not match the current Part 5 configuration.')
real_test_df = canonicalize_frame(real_raw)
print('Synthetic:', synthetic_df.shape, '| held-out real test:', real_test_df.shape)
display(synthetic_df.head())

# ## 3. Define marginal and pairwise metrics
#
# For each field, the fixed public category domain is used to align probability vectors (including zero-count categories). JSD is reported in bits by using base 2. TVD ranges from 0 (identical empirical marginals) to 1 (no category overlap).

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


def category_probability(frame, column):
    counts = frame[column].value_counts().reindex(DOMAINS[column], fill_value=0).to_numpy(dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)

marginal_rows = []
for col in ALL_COLS:
    p_real = category_probability(real_test_df, col)
    p_syn = category_probability(synthetic_df, col)
    tvd = 0.5 * float(np.abs(p_real - p_syn).sum())
    jsd = float(jensenshannon(p_real, p_syn, base=2) ** 2)
    marginal_rows.append({
        'variable': col,
        'real_test_n': int(len(real_test_df)),
        'synthetic_n': int(len(synthetic_df)),
        'total_variation_distance': tvd,
        'jensen_shannon_divergence_bits': jsd,
        'real_unique_categories': int(real_test_df[col].nunique()),
        'synthetic_unique_categories': int(synthetic_df[col].nunique()),
    })
marginal_df = pd.DataFrame(marginal_rows).sort_values('total_variation_distance', ascending=False)
display(marginal_df)

def corrected_cramers_v(a, b):
    """Bias-corrected Cramer's V for two categorical vectors; returns NaN if undefined."""
    table = pd.crosstab(pd.Series(a, dtype='category'), pd.Series(b, dtype='category'))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    obs = table.to_numpy(dtype=float)
    n = obs.sum()
    if n <= 1:
        return np.nan
    try:
        chi2 = chi2_contingency(obs, correction=False)[0]
    except ValueError:
        return np.nan
    phi2 = chi2 / n
    r, k = obs.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2_corr / denom))

PAIR_COLS = ALL_COLS  # Includes DIABETE4 as a field in the association analysis.
real_assoc = pd.DataFrame(np.eye(len(PAIR_COLS)), index=PAIR_COLS, columns=PAIR_COLS)
syn_assoc = pd.DataFrame(np.eye(len(PAIR_COLS)), index=PAIR_COLS, columns=PAIR_COLS)
pair_rows = []
for i, a in enumerate(PAIR_COLS):
    for b in PAIR_COLS[i+1:]:
        vr = corrected_cramers_v(real_test_df[a], real_test_df[b])
        vs = corrected_cramers_v(synthetic_df[a], synthetic_df[b])
        real_assoc.loc[a, b] = real_assoc.loc[b, a] = vr
        syn_assoc.loc[a, b] = syn_assoc.loc[b, a] = vs
        pair_rows.append({'variable_1': a, 'variable_2': b, 'real_test_cramers_v': vr,
                          'synthetic_cramers_v': vs,
                          'absolute_association_difference': abs(vr-vs) if np.isfinite(vr) and np.isfinite(vs) else np.nan})
pairwise_df = pd.DataFrame(pair_rows)
display(pairwise_df.sort_values('absolute_association_difference', ascending=False).head(15))

# ## 4. Save evaluation tables and calculate summary statistics

valid_pair_diffs = pairwise_df['absolute_association_difference'].dropna()
summary = {
    'synthetic_model_label': MODEL_LABEL,
    'real_comparison_split': 'Part 2 held-out test split',
    'n_real_test': int(len(real_test_df)),
    'n_synthetic': int(len(synthetic_df)),
    'mean_tvd_across_fields': float(marginal_df['total_variation_distance'].mean()),
    'median_tvd_across_fields': float(marginal_df['total_variation_distance'].median()),
    'mean_jsd_bits_across_fields': float(marginal_df['jensen_shannon_divergence_bits'].mean()),
    'mean_absolute_pairwise_cramers_v_difference': float(valid_pair_diffs.mean()) if len(valid_pair_diffs) else None,
    'number_of_pairwise_associations_compared': int(len(valid_pair_diffs)),
    'interpretation': 'Lower values indicate closer empirical similarity; these are descriptive fidelity measures, not formal privacy or clinical-validity guarantees.'
}
MARGINAL_PATH = METRICS_DIR / f'part10_marginal_fidelity_{MODEL_LABEL}.csv'
PAIRWISE_PATH = METRICS_DIR / f'part10_pairwise_fidelity_{MODEL_LABEL}.csv'
SUMMARY_PATH = METRICS_DIR / f'part10_fidelity_summary_{MODEL_LABEL}.json'
ASSOC_REAL_PATH = METRICS_DIR / f'part10_real_association_{MODEL_LABEL}.csv'
ASSOC_SYN_PATH = METRICS_DIR / f'part10_synthetic_association_{MODEL_LABEL}.csv'
marginal_df.to_csv(MARGINAL_PATH, index=False)
pairwise_df.to_csv(PAIRWISE_PATH, index=False)
real_assoc.to_csv(ASSOC_REAL_PATH)
syn_assoc.to_csv(ASSOC_SYN_PATH)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))

# ## 5. Create report-ready fidelity graphs

# Plot the six variables with the largest marginal TVD.
top_vars = marginal_df.head(6)['variable'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), top_vars):
    p_real = category_probability(real_test_df, col)
    p_syn = category_probability(synthetic_df, col)
    x = np.arange(len(DOMAINS[col]))
    width = 0.40
    ax.bar(x - width/2, p_real, width, label='Real test')
    ax.bar(x + width/2, p_syn, width, label='Synthetic')
    ax.set_title(col)
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS[col], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Proportion')
    ax.grid(axis='y', alpha=0.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Marginal category distributions: real held-out vs synthetic')
fig.tight_layout()
MARGINAL_PLOT = PLOTS_DIR / f'part10_marginal_comparison_{MODEL_LABEL}.png'
fig.savefig(MARGINAL_PLOT, dpi=180, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, matrix, title in [(axes[0], real_assoc, 'Real test associations'),
                          (axes[1], syn_assoc, 'Synthetic associations'),
                          (axes[2], (real_assoc - syn_assoc).abs(), 'Absolute difference')]:
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1 if title != 'Absolute difference' else None, aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xticks(range(len(PAIR_COLS))); ax.set_xticklabels(PAIR_COLS, rotation=90, fontsize=6)
    ax.set_yticks(range(len(PAIR_COLS))); ax.set_yticklabels(PAIR_COLS, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
ASSOC_PLOT = PLOTS_DIR / f'part10_association_comparison_{MODEL_LABEL}.png'
fig.savefig(ASSOC_PLOT, dpi=180, bbox_inches='tight')
plt.show()
print('Saved:', MARGINAL_PLOT)
print('Saved:', ASSOC_PLOT)

# ## 6. How to interpret the evaluation
#
# - A lower TVD/JSD means the marginal category proportions are closer. Review the per-variable table; an average can hide a large mismatch in an important field.
# - A smaller pairwise Cramér's V difference means the synthetic table better reproduces the strength of pairwise categorical associations. Pairwise metrics do not measure all higher-order dependencies.
# - Fidelity is not the same as utility. Part 11 assesses a downstream prediction task using real held-out respondents.
# - High fidelity is not evidence of privacy. Part 12 performs separate exact-match, nearest-neighbour and distance-based membership-risk diagnostics; Part 8 is the source for the formal DP accounting result.
# - This project uses public self-reported survey data, not hospital EHR data. No clinical claim should be made from these synthetic records.
#
# ## References
#
# 1. Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability distributions. *IEEE Transactions on Information Theory*, 49(7), 1858–1860.
# 2. Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42, 323–328.
# 3. CDC. 2025 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2025.html
