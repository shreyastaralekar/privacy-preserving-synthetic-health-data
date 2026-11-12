#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part11_DP_Medical_TSTR_TRTR_Utility.ipynb.

Run from the project root with: python scripts/part11_tstr_trtr_utility.py
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
print("Part 11 – Part11_DP_Medical_TSTR_TRTR_Utility")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 11 – Downstream Predictive Utility: TSTR vs TRTR
#
# **TSTR** means *Train on Synthetic, Test on Real*. **TRTR** means *Train on Real, Test on Real*. Both models are evaluated on the same untouched real test respondents. This experiment asks whether relationships learned from the synthetic table preserve information useful for one clearly defined prediction task.
#
# ## Prediction task and target handling
#
# The selected response is `DIABETE4`. For a binary task, code `1` is treated as diabetes reported and code `3` as no diabetes reported. Codes `2`, `4`, `7` and `9` are excluded from the primary binary evaluation because they represent pregnancy-only diabetes, prediabetes/borderline diabetes, don't know, or refused responses in the survey coding. These definitions are based on the BRFSS codebook and should be confirmed against the exact 2025 documentation before formal publication.
#
# The model is a logistic-regression classifier with one-hot encoding for all categorical predictors. This notebook reports ROC-AUC, PR-AUC, balanced accuracy, precision, recall, F1 and confusion-matrix counts. Results are utility indicators for this defined task, not evidence of diagnosis or clinical readiness.

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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_curve)

# ## 2. Load real and synthetic data

MODEL_LABEL = 'dp_eps_3p0'  # Must match the model generated in Part 9.
SYNTHETIC_PATH = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}.csv'
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
for p in [SYNTHETIC_PATH, TRAIN_PATH, TEST_PATH]:
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Complete Parts 2 and 9 first.')
synthetic_df = canonicalize_frame(pd.read_csv(SYNTHETIC_PATH, dtype=str, keep_default_na=False))
synthetic_meta_path = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}_metadata.json'
if synthetic_meta_path.exists():
    synthetic_meta = json.loads(synthetic_meta_path.read_text(encoding='utf-8'))
    if synthetic_meta.get('data_fingerprint') != config.get('data_fingerprint'):
        raise ValueError('Synthetic data metadata fingerprint does not match the current Part 5 configuration.')
real_train_df = canonicalize_frame(pd.read_csv(TRAIN_PATH))
real_test_df = canonicalize_frame(pd.read_csv(TEST_PATH))
print('Synthetic:', synthetic_df.shape, '| real train:', real_train_df.shape, '| real test:', real_test_df.shape)

# ## 3. Create a consistent binary diabetes task
#
# Rows with ambiguous or out-of-scope target values are not assigned to either class. The exact same rule is applied to synthetic training, real training and real test data.

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

def to_binary_task(frame):
    data = frame[ALL_COLS].copy()
    y_raw = data[TARGET_COL].astype(str)
    valid = y_raw.isin(['1', '3'])
    X = data.loc[valid, FEATURE_COLS].astype(str).copy()
    y = (y_raw.loc[valid] == '1').astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True), int((~valid).sum())

X_syn, y_syn, dropped_syn = to_binary_task(synthetic_df)
X_real_train, y_real_train, dropped_real_train = to_binary_task(real_train_df)
X_real_test, y_real_test, dropped_real_test = to_binary_task(real_test_df)
if y_syn.nunique() < 2 or y_real_train.nunique() < 2 or y_real_test.nunique() < 2:
    raise ValueError('The binary task needs both target classes in synthetic train, real train and real test data. Inspect DIABETE4 distribution or generate a larger synthetic dataset.')
print('Usable rows | synthetic:', len(y_syn), '| real train:', len(y_real_train), '| real test:', len(y_real_test))
print('Excluded ambiguous target rows | synthetic:', dropped_syn, '| real train:', dropped_real_train, '| real test:', dropped_real_test)
print('Positive prevalence | synthetic:', round(float(y_syn.mean()), 4), '| real train:', round(float(y_real_train.mean()), 4), '| real test:', round(float(y_real_test.mean()), 4))

# ## 4. Fit TSTR and TRTR on equal-sized training samples
#
# To avoid giving one method a larger training sample, both arms use the same number of records, capped at 20,000 for a practical Colab run. The real test split is not used during fitting or hyperparameter tuning.

MAX_TRAIN_ROWS = 20_000
N_COMPARE = min(len(y_syn), len(y_real_train), MAX_TRAIN_ROWS)
if N_COMPARE < 100:
    warnings.warn('Fewer than 100 usable training records; performance estimates may be unstable.')

syn_idx = np.random.default_rng(SEED).choice(len(y_syn), size=N_COMPARE, replace=False)
real_idx = np.random.default_rng(SEED + 1).choice(len(y_real_train), size=N_COMPARE, replace=False)
X_syn_fit, y_syn_fit = X_syn.iloc[syn_idx], y_syn.iloc[syn_idx]
X_real_fit, y_real_fit = X_real_train.iloc[real_idx], y_real_train.iloc[real_idx]

# One-hot encoder keeps categorical values nominal rather than treating integer codes as continuous.
def build_classifier():
    preprocessing = ColumnTransformer([('categorical', OneHotEncoder(handle_unknown='ignore'), FEATURE_COLS)], remainder='drop')
    classifier = LogisticRegression(max_iter=1500, class_weight='balanced', solver='liblinear', random_state=SEED)
    return Pipeline([('preprocess', preprocessing), ('classifier', classifier)])

model_tstr = build_classifier()
model_trtr = build_classifier()
model_tstr.fit(X_syn_fit, y_syn_fit)
model_trtr.fit(X_real_fit, y_real_fit)
print('TSTR and TRTR models fitted using', N_COMPARE, 'records each.')

# ## 5. Evaluate both models on the same real test set

def evaluate_model(label, fitted_model, X_test, y_test):
    prob = fitted_model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'training_regime': label,
        'training_rows': int(N_COMPARE),
        'real_test_rows': int(len(y_test)),
        'roc_auc': float(roc_auc_score(y_test, prob)),
        'pr_auc_average_precision': float(average_precision_score(y_test, prob)),
        'accuracy': float(accuracy_score(y_test, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall_sensitivity': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else float('nan'),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        '_probabilities': prob,
        '_confusion_matrix': cm,
    }

res_tstr = evaluate_model('TSTR', model_tstr, X_real_test, y_real_test)
res_trtr = evaluate_model('TRTR', model_trtr, X_real_test, y_real_test)
result_rows = []
for r in [res_tstr, res_trtr]:
    result_rows.append({k:v for k,v in r.items() if not k.startswith('_')})
results_df = pd.DataFrame(result_rows)
display(results_df)

# ## 6. Save utility tables and plots

RESULT_PATH = METRICS_DIR / f'part11_tstr_trtr_metrics_{MODEL_LABEL}.csv'
DETAIL_PATH = METRICS_DIR / f'part11_tstr_trtr_details_{MODEL_LABEL}.json'
results_df.to_csv(RESULT_PATH, index=False)
details = {
    'model_label': MODEL_LABEL,
    'target': TARGET_COL,
    'positive_class': 'DIABETE4 code 1 (Yes)',
    'negative_class': 'DIABETE4 code 3 (No)',
    'excluded_target_codes': ['2', '4', '7', '9'],
    'training_rows_per_arm': int(N_COMPARE),
    'evaluation_split': 'Part 2 held-out real test data',
    'features': FEATURE_COLS,
    'metrics_csv': RESULT_PATH.name,
    'interpretation': 'A descriptive TSTR vs TRTR benchmark on one binary classification task; not clinical validation.'
}
DETAIL_PATH.write_text(json.dumps(details, indent=2), encoding='utf-8')

fig, ax = plt.subplots(figsize=(9, 5))
metric_names = ['roc_auc', 'pr_auc_average_precision', 'balanced_accuracy', 'f1']
x = np.arange(len(metric_names)); width = 0.36
ax.bar(x - width/2, [res_tstr[m] for m in metric_names], width, label='TSTR')
ax.bar(x + width/2, [res_trtr[m] for m in metric_names], width, label='TRTR')
ax.set_xticks(x); ax.set_xticklabels(['ROC-AUC', 'PR-AUC', 'Balanced accuracy', 'F1'])
ax.set_ylim(0, 1); ax.set_ylabel('Score'); ax.set_title('Downstream utility: train on synthetic vs train on real')
ax.grid(axis='y', alpha=0.25); ax.legend()
fig.tight_layout()
PLOT_PATH = PLOTS_DIR / f'part11_tstr_trtr_comparison_{MODEL_LABEL}.png'
fig.savefig(PLOT_PATH, dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
for label, result in [('TSTR', res_tstr), ('TRTR', res_trtr)]:
    fpr, tpr, _ = roc_curve(y_real_test, result['_probabilities'])
    ax.plot(fpr, tpr, label=f"{label} (AUC={result['roc_auc']:.3f})")
ax.plot([0,1], [0,1], linestyle='--', linewidth=1)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('ROC curves on the common real test split'); ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout()
ROC_PATH = PLOTS_DIR / f'part11_tstr_trtr_roc_{MODEL_LABEL}.png'
fig.savefig(ROC_PATH, dpi=180, bbox_inches='tight')
plt.show()
print('Saved metrics:', RESULT_PATH)
print('Saved plots:', PLOT_PATH, 'and', ROC_PATH)

# ## 7. Interpretation and limitations
#
# - TRTR is the real-data reference for this experiment. TSTR measures how well the synthetic training table supports the same prediction task on real held-out rows.
# - A TSTR score close to TRTR suggests that some task-relevant relationships have been preserved. A gap may reflect marginal imbalance, loss of associations, sampling noise, or the chosen classifier.
# - Class balancing in logistic regression changes the decision model; report the same procedure for both training arms.
# - The primary task excludes BRFSS target codes `3`, `4`, `7` and `9`. Report the excluded proportions so the evaluation population is clear.
# - This is not a diagnostic model, clinical deployment test, causal analysis, or fairness audit. It evaluates one prediction target and one classifier family. Consider subgroup performance and alternative models in future extensions.
# - This metric does not prove privacy. Read Part 8 and Part 12 for the separate privacy analyses.
#
# ## References
#
# 1. Jordon, J., Yoon, J., & van der Schaar, M. (2019). **PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees.** ICLR.
# 2. CDC. 2025 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2025.html
# 3. scikit-learn documentation. https://scikit-learn.org/stable/
