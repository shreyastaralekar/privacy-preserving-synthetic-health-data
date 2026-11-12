#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part13_DP_Medical_Utility_Privacy_Tradeoff.ipynb.

Run from the project root with: python scripts/part13_utility_privacy_tradeoff.py
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
print("Part 13 – Part13_DP_Medical_Utility_Privacy_Tradeoff")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 13 – Integrated Utility–Privacy Trade-off Analysis
#
# **Purpose:** bring together the measured outputs from Parts 8, 10, 11 and 12 for each generated dataset/model label. This notebook compares privacy accounting, statistical fidelity, downstream utility and empirical disclosure-risk diagnostics in one reproducible analysis.
#
# **Inputs:** `privacy_accounting_verification.csv`, `part10_fidelity_summary_*.json`, `part11_tstr_trtr_metrics_*.csv`, `part12_privacy_audit_summary_*.json`, and matching synthetic-data metadata.
#
# **Outputs:** a consolidated CSV/JSON/Markdown report, Pareto-frontier flags, and comparison plots in the shared Google Drive project folder.
#
# > No results are fabricated. Missing experiment files remain missing/NaN and are reported as pending. Empirical distance or membership tests are not substitutes for a formal differential-privacy guarantee.

# ## 1. Install dependencies and mount persistent Google Drive

# [Notebook install command moved to requirements.txt]

from pathlib import Path
import json, math, warnings, re, os, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

SEED = 42
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
TRADEOFF_DIR = ARTIFACT_DIR / 'tradeoff_analysis'
for folder in [PROJECT_ROOT, DATA_DIR, PLOTS_DIR, CHECKPOINT_DIR, MODEL_DIR,
               METRICS_DIR, ARTIFACT_DIR, SYNTHETIC_DIR, TRADEOFF_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = DATA_DIR / 'ddpm_config.json'
SCHEMA_PATH = DATA_DIR / 'schema_manifest.json'
if not CONFIG_PATH.exists() or not SCHEMA_PATH.exists():
    raise FileNotFoundError('Run Parts 1–5 first. Required: data/ddpm_config.json and data/schema_manifest.json.')
config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
DATA_FINGERPRINT = str(config.get('data_fingerprint', ''))
print('Storage:', STORAGE_MODE)
print('Project root:', PROJECT_ROOT)
print('Dataset:', schema.get('dataset', 'Unspecified'))
print('Training data fingerprint:', DATA_FINGERPRINT[:16] + '...')

# ## 2. Discover model labels and choose the comparison set
#
# A model label is discovered from a saved `synthetic_<label>.csv` file. The default compares every discovered label. You may restrict the analysis to selected labels, but do not compare differently-preprocessed datasets or mismatched feature schemas.

synthetic_files = sorted(SYNTHETIC_DIR.glob('synthetic_*.csv'))
DISCOVERED_LABELS = [p.stem[len('synthetic_'):] for p in synthetic_files]
if not DISCOVERED_LABELS:
    raise FileNotFoundError('No synthetic data files found. Complete Part 9 for at least one model first.')

# Default: compare every available synthetic release.
# Example manual selection: MODEL_LABELS = ['dp_eps_1p0', 'dp_eps_3p0', 'dp_eps_8p0']
MODEL_LABELS = DISCOVERED_LABELS.copy()
print('Discovered labels:', DISCOVERED_LABELS)
print('Selected labels:', MODEL_LABELS)

# ## 3. Load each experiment artifact without filling missing values

def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        warnings.warn(f'Could not parse {path.name}: {exc}')
        return None


def first_row(frame, name):
    if frame is None or frame.empty or 'training_regime' not in frame.columns:
        return None
    rows = frame.loc[frame['training_regime'].astype(str).str.upper() == name]
    return None if rows.empty else rows.iloc[0].to_dict()

verification_path = METRICS_DIR / 'privacy_accounting_verification.csv'
verification_df = pd.read_csv(verification_path) if verification_path.exists() else pd.DataFrame()
if not verification_df.empty and 'target_epsilon' in verification_df.columns:
    verification_df['target_epsilon'] = pd.to_numeric(verification_df['target_epsilon'], errors='coerce')

records = []
for label in MODEL_LABELS:
    row = {
        'model_label': label,
        'model_kind': 'non_private' if label == 'non_private' else ('dp' if label.startswith('dp_eps_') else 'unknown'),
        'synthetic_rows': np.nan,
        'formal_dp_verified': False,
        'dp_accounting_status': 'NOT_APPLICABLE' if label == 'non_private' else 'NOT_VERIFIED',
        'target_epsilon': np.nan,
        'achieved_epsilon': np.nan,
        'delta': np.nan,
        'mean_tvd_across_fields': np.nan,
        'mean_jsd_bits_across_fields': np.nan,
        'mean_absolute_pairwise_cramers_v_difference': np.nan,
        'tstr_roc_auc': np.nan,
        'trtr_roc_auc': np.nan,
        'roc_auc_retention': np.nan,
        'tstr_pr_auc': np.nan,
        'trtr_pr_auc': np.nan,
        'pr_auc_retention': np.nan,
        'tstr_balanced_accuracy': np.nan,
        'trtr_balanced_accuracy': np.nan,
        'tstr_f1': np.nan,
        'trtr_f1': np.nan,
        'exact_match_rate': np.nan,
        'synthetic_duplicate_row_rate': np.nan,
        'synthetic_to_training_dcr_median': np.nan,
        'distance_based_membership_attack_auc': np.nan,
        'fidelity_available': False,
        'utility_available': False,
        'privacy_audit_available': False,
        'synthetic_metadata_available': False,
    }

    syn_meta_path = SYNTHETIC_DIR / f'synthetic_{label}_metadata.json'
    syn_meta = read_json(syn_meta_path)
    if syn_meta is not None:
        row['synthetic_metadata_available'] = True
        row['synthetic_rows'] = syn_meta.get('rows_generated', np.nan)
        if syn_meta.get('data_fingerprint') and DATA_FINGERPRINT and syn_meta.get('data_fingerprint') != DATA_FINGERPRINT:
            raise ValueError(f'{label}: synthetic metadata fingerprint does not match the current Part 5 configuration.')

    fidelity = read_json(METRICS_DIR / f'part10_fidelity_summary_{label}.json')
    if fidelity is not None:
        row['fidelity_available'] = True
        for key in ['n_synthetic', 'mean_tvd_across_fields', 'mean_jsd_bits_across_fields',
                    'mean_absolute_pairwise_cramers_v_difference']:
            if key in fidelity:
                row['synthetic_rows'] = fidelity.get('n_synthetic', row['synthetic_rows']) if key == 'n_synthetic' else row['synthetic_rows']
                row[key] = fidelity.get(key, np.nan)

    utility_path = METRICS_DIR / f'part11_tstr_trtr_metrics_{label}.csv'
    if utility_path.exists():
        utility_df = pd.read_csv(utility_path)
        tstr = first_row(utility_df, 'TSTR')
        trtr = first_row(utility_df, 'TRTR')
        if tstr and trtr:
            row['utility_available'] = True
            mapping = {
                'roc_auc': ('tstr_roc_auc', 'trtr_roc_auc', 'roc_auc_retention'),
                'pr_auc_average_precision': ('tstr_pr_auc', 'trtr_pr_auc', 'pr_auc_retention'),
                'balanced_accuracy': ('tstr_balanced_accuracy', 'trtr_balanced_accuracy', None),
                'f1': ('tstr_f1', 'trtr_f1', None),
            }
            for metric, (tkey, rkey, retention_key) in mapping.items():
                tv = pd.to_numeric(pd.Series([tstr.get(metric)]), errors='coerce').iloc[0]
                rv = pd.to_numeric(pd.Series([trtr.get(metric)]), errors='coerce').iloc[0]
                row[tkey], row[rkey] = tv, rv
                if retention_key:
                    row[retention_key] = float(tv / rv) if pd.notna(tv) and pd.notna(rv) and rv != 0 else np.nan

    audit = read_json(METRICS_DIR / f'part12_privacy_audit_summary_{label}.json')
    if audit is not None:
        row['privacy_audit_available'] = True
        for key in ['exact_match_rate', 'synthetic_duplicate_row_rate',
                    'synthetic_to_training_dcr_median', 'distance_based_membership_attack_auc']:
            row[key] = audit.get(key, np.nan)

    # Formal DP status is accepted only when synthetic metadata, Part 7 run metadata,
    # and the Part 8 recomputation agree for the selected model and data fingerprint.
    if row['model_kind'] == 'dp':
        tag = label.replace('dp_eps_', '')
        run_meta = read_json(METRICS_DIR / f'dp_run_eps_{tag}.json')
        if run_meta is not None:
            row['target_epsilon'] = run_meta.get('target_epsilon', np.nan)
            if run_meta.get('data_fingerprint') != DATA_FINGERPRINT:
                row['dp_accounting_status'] = 'FINGERPRINT_MISMATCH'
            elif not verification_df.empty:
                eps_value = pd.to_numeric(pd.Series([run_meta.get('target_epsilon')]), errors='coerce').iloc[0]
                matches = verification_df[np.isclose(verification_df['target_epsilon'], eps_value, equal_nan=False)]
                if len(matches):
                    v = matches.iloc[0]
                    row['achieved_epsilon'] = v.get('recomputed_epsilon', np.nan)
                    row['delta'] = v.get('delta', np.nan)
                    check_ok = str(v.get('verification', '')).upper() == 'PASS'
                    target_ok = str(v.get('target_budget_status', '')).upper() == 'WITHIN_TARGET'
                    meta_ok = syn_meta is not None and bool(syn_meta.get('formal_dp_claim', False))
                    row['formal_dp_verified'] = bool(check_ok and target_ok and meta_ok)
                    row['dp_accounting_status'] = 'VERIFIED' if row['formal_dp_verified'] else 'CHECK_FAILED_OR_SYNTHETIC_NOT_VERIFIED'
                else:
                    row['dp_accounting_status'] = 'NO_MATCHING_PART8_ROW'
            else:
                row['dp_accounting_status'] = 'PART8_VERIFICATION_MISSING'
        else:
            row['dp_accounting_status'] = 'PART7_METADATA_MISSING'

    records.append(row)

comparison = pd.DataFrame(records)
# Ensure numeric metrics remain numeric for plotting and downstream exports.
num_cols = [c for c in comparison.columns if c not in ['model_label','model_kind','dp_accounting_status']]
for c in num_cols:
    comparison[c] = pd.to_numeric(comparison[c], errors='coerce')
display(comparison)

# ## 4. Define comparable privacy–utility criteria
#
# The analysis keeps the dimensions separate instead of hiding them inside an arbitrary weighted score:
#
# - **Formal privacy:** lower verified achieved ε is preferable at the same δ and under the same privacy-unit definition. Non-private models have no finite DP ε.
# - **Marginal fidelity:** lower mean total-variation distance (TVD) and Jensen–Shannon divergence (JSD) indicate closer empirical category distributions.
# - **Association fidelity:** lower absolute difference in pairwise Cramér's V indicates closer measured associations.
# - **Downstream utility:** TSTR is compared with TRTR on the same real held-out test split. Ratios are descriptive retention indicators, not clinical-performance thresholds.
# - **Empirical audit:** exact matches, duplicate rates, DCR and distance-based membership AUC are diagnostics only, not formal privacy guarantees.
#
# The Pareto frontier below uses only verified DP models with all three primary dimensions available: achieved ε (minimise), mean TVD (minimise), and TSTR ROC-AUC (maximise). It does not declare one model universally best.

# Add Pareto frontier flags among complete, accountant-verified DP candidates only.
comparison['pareto_efficient'] = False
valid_mask = (
    (comparison['model_kind'] == 'dp') &
    comparison['formal_dp_verified'].fillna(False) &
    comparison['achieved_epsilon'].notna() &
    comparison['mean_tvd_across_fields'].notna() &
    comparison['tstr_roc_auc'].notna()
)
valid_idx = comparison.index[valid_mask].tolist()
for i in valid_idx:
    a = comparison.loc[i]
    dominated = False
    for j in valid_idx:
        if i == j:
            continue
        b = comparison.loc[j]
        no_worse = (
            b['achieved_epsilon'] <= a['achieved_epsilon'] and
            b['mean_tvd_across_fields'] <= a['mean_tvd_across_fields'] and
            b['tstr_roc_auc'] >= a['tstr_roc_auc']
        )
        strictly_better = (
            b['achieved_epsilon'] < a['achieved_epsilon'] or
            b['mean_tvd_across_fields'] < a['mean_tvd_across_fields'] or
            b['tstr_roc_auc'] > a['tstr_roc_auc']
        )
        if no_worse and strictly_better:
            dominated = True
            break
    comparison.loc[i, 'pareto_efficient'] = not dominated

TRADEOFF_CSV = TRADEOFF_DIR / 'part13_tradeoff_comparison.csv'
TRADEOFF_JSON = TRADEOFF_DIR / 'part13_tradeoff_summary.json'
comparison.to_csv(TRADEOFF_CSV, index=False)
print('Saved consolidated comparison:', TRADEOFF_CSV)
print('Verified DP labels:', comparison.loc[comparison['formal_dp_verified'], 'model_label'].tolist())
print('Pareto-efficient labels:', comparison.loc[comparison['pareto_efficient'], 'model_label'].tolist())

# ## 5. Generate trade-off plots

def save_scatter(frame, xcol, ycol, title, xlabel, ylabel, filename, verified_only=False):
    plot_df = frame.dropna(subset=[xcol, ycol]).copy()
    if verified_only:
        plot_df = plot_df[plot_df['formal_dp_verified'].fillna(False)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_df.empty:
        ax.text(0.5, 0.5, 'No complete verified measurements available yet',
                ha='center', va='center', transform=ax.transAxes)
    else:
        for _, r in plot_df.iterrows():
            ax.scatter(r[xcol], r[ycol], s=75)
            ax.annotate(str(r['model_label']), (r[xcol], r[ycol]), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
        if 'pareto_efficient' in plot_df.columns and plot_df['pareto_efficient'].any() and ycol == 'mean_tvd_across_fields':
            front = plot_df[plot_df['pareto_efficient']].sort_values(xcol)
            ax.plot(front[xcol], front[ycol], linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.show()
    return path

plot_paths = []
plot_paths.append(save_scatter(comparison, 'achieved_epsilon', 'mean_tvd_across_fields',
    'Formal privacy budget vs marginal fidelity', 'Verified achieved epsilon (lower is stronger)',
    'Mean marginal TVD (lower is closer)', 'part13_epsilon_vs_tvd.png', verified_only=True))
plot_paths.append(save_scatter(comparison, 'achieved_epsilon', 'tstr_roc_auc',
    'Formal privacy budget vs downstream TSTR utility', 'Verified achieved epsilon',
    'TSTR ROC-AUC on held-out real test set (higher is better)', 'part13_epsilon_vs_tstr_auc.png', verified_only=True))
plot_paths.append(save_scatter(comparison, 'achieved_epsilon', 'roc_auc_retention',
    'Privacy budget vs TSTR/TRTR ROC-AUC retention', 'Verified achieved epsilon',
    'TSTR ROC-AUC divided by TRTR ROC-AUC', 'part13_epsilon_vs_utility_retention.png', verified_only=True))
plot_paths.append(save_scatter(comparison, 'achieved_epsilon', 'distance_based_membership_attack_auc',
    'Privacy budget vs empirical distance-attack AUC', 'Verified achieved epsilon',
    'Distance-based membership AUC (near 0.5: this attack near chance)',
    'part13_epsilon_vs_membership_auc.png', verified_only=True))

# A compact metric-availability figure makes incomplete runs visible rather than hiding them.
availability_cols = ['fidelity_available','utility_available','privacy_audit_available','formal_dp_verified']
if not comparison.empty:
    avail = comparison.set_index('model_label')[availability_cols].astype(int)
    fig, ax = plt.subplots(figsize=(9, max(2.8, 0.65 * len(avail))))
    im = ax.imshow(avail.to_numpy(), aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(avail)), avail.index)
    ax.set_xticks(range(len(availability_cols)), ['Fidelity','TSTR/TRTR','Audit','DP verified'])
    ax.set_title('Experiment artifact completeness by model')
    for y in range(avail.shape[0]):
        for x in range(avail.shape[1]):
            ax.text(x, y, 'Yes' if avail.iloc[y, x] else 'Pending', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, ticks=[0,1], label='Availability')
    fig.tight_layout()
    completeness_path = PLOTS_DIR / 'part13_artifact_completeness.png'
    fig.savefig(completeness_path, dpi=180, bbox_inches='tight')
    plt.show()
    plot_paths.append(completeness_path)

# ## 6. Save a machine-readable summary and human-readable report

verified = comparison[comparison['formal_dp_verified'].fillna(False)]
pareto_labels = comparison.loc[comparison['pareto_efficient'], 'model_label'].tolist()
summary = {
    'analysis': 'Part 13 integrated utility-privacy trade-off analysis',
    'dataset': schema.get('dataset', 'CDC 2025 BRFSS'),
    'data_fingerprint': DATA_FINGERPRINT,
    'model_labels_compared': MODEL_LABELS,
    'model_count': int(len(comparison)),
    'verified_dp_model_count': int(len(verified)),
    'verified_dp_model_labels': verified['model_label'].tolist(),
    'pareto_efficient_verified_dp_labels': pareto_labels,
    'pareto_dimensions': ['achieved_epsilon_minimize','mean_tvd_across_fields_minimize','tstr_roc_auc_maximize'],
    'missing_metric_counts': {c: int(comparison[c].isna().sum()) for c in [
        'achieved_epsilon','mean_tvd_across_fields','tstr_roc_auc','exact_match_rate','distance_based_membership_attack_auc']},
    'interpretation_limitations': [
        'Empirical fidelity and privacy-risk audits are not formal DP guarantees.',
        'Only Part 8 accountant-verified models whose Part 9 metadata also marks formal_dp_claim true are treated as verified.',
        'The Pareto frontier is descriptive and uses only verified DP runs with all three primary metrics available.',
        'TSTR/TRTR metrics are one downstream binary classification benchmark, not clinical validation.',
        'When multiple separately trained DP models or datasets are released, privacy losses must be composed.'
    ],
    'comparison_csv': TRADEOFF_CSV.name,
    'plots': [p.name for p in plot_paths],
}
TRADEOFF_JSON.write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

md_lines = [
    '# Part 13 – Integrated Utility–Privacy Trade-off Results',
    '',
    f"- Dataset: {summary['dataset']}",
    f"- Model labels compared: {len(MODEL_LABELS)}",
    f"- Accountant-verified DP models: {len(verified)}",
    f"- Pareto-efficient verified DP models: {', '.join(pareto_labels) if pareto_labels else 'None identified / not enough complete measurements'}",
    '',
    '## Consolidated measurements',
    '',
    '| Model | DP status | Achieved ε | Mean TVD ↓ | TSTR ROC-AUC ↑ | TSTR/TRTR AUC ratio | Exact-match rate | Distance-attack AUC | Pareto |',
    '|---|---|---:|---:|---:|---:|---:|---:|---|'
]
for _, r in comparison.iterrows():
    def fmt(v, digits=4): return '—' if pd.isna(v) else f'{float(v):.{digits}f}'
    md_lines.append('| ' + ' | '.join([
        str(r['model_label']), str(r['dp_accounting_status']), fmt(r['achieved_epsilon']),
        fmt(r['mean_tvd_across_fields']), fmt(r['tstr_roc_auc']), fmt(r['roc_auc_retention']),
        fmt(r['exact_match_rate']), fmt(r['distance_based_membership_attack_auc']),
        'Yes' if bool(r['pareto_efficient']) else 'No'
    ]) + ' |')
md_lines += [
    '', '## Interpretation rules',
    '',
    '- Lower verified ε means a stronger formal privacy bound at the reported δ, subject to the accountant assumptions and correct mechanism implementation.',
    '- Lower TVD/JSD and lower pairwise association differences indicate closer empirical distributions.',
    '- TSTR is interpreted against TRTR evaluated on the same held-out real test split. Ratios are not universal acceptability thresholds.',
    '- Empirical attacks and DCR are diagnostics. A near-chance attack does not establish absence of privacy risk.',
    '- The Pareto set identifies non-dominated candidates on the selected dimensions; it is not a single best-model ranking.',
    '', '## Missing work',
    '',
    'A dash (—) means the corresponding upstream artifact is not available or was not measurable. Complete the relevant Part 8, 10, 11 or 12 notebook and rerun this analysis.',
    '', '## Privacy release note',
    '',
    'If several separately trained models trained on the same respondents are released together, their privacy losses compose. Report the release policy and the applicable combined privacy bound. The non-private model has no formal DP guarantee.',
]
TRADEOFF_MD = TRADEOFF_DIR / 'part13_tradeoff_report.md'
TRADEOFF_MD.write_text('\n'.join(md_lines), encoding='utf-8')
print('Saved summary:', TRADEOFF_JSON)
print('Saved report:', TRADEOFF_MD)
display(Markdown(TRADEOFF_MD.read_text(encoding='utf-8')))

# ## 7. Part 13 completion checklist
#
# - [ ] Synthetic outputs exist for every model intended for comparison.
# - [ ] Part 8 accountant verification completed for each DP model.
# - [ ] Parts 10, 11 and 12 completed using the same model labels and held-out split.
# - [ ] Part 13 CSV/JSON/Markdown and all plots saved under `artifacts/tradeoff_analysis/`, `metrics/` and `plots/`.
# - [ ] Any model release decision accounts for composition if multiple DP models are disclosed.
#
# **Next:** run Part 14 to build the final report and presentation from these saved, measured artifacts.
