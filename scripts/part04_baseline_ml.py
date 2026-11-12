#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part4_DP_Medical_Baseline_ML.ipynb.

Run from the project root with: python scripts/part04_baseline_ml.py
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
print("Part 04 – Part4_DP_Medical_Baseline_ML")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 4 – Baseline Machine Learning Models
#
# **Goal:** Establish a real-data baseline for the downstream diabetes classification task before training synthetic-data generators.
#
# **Task:** Predict clear diabetes yes/no responses from the selected BRFSS health and demographic features. `DIABETE4=1` is positive (yes) and `DIABETE4=3` is negative (no). Other responses are excluded only from this binary classification experiment; they remain in the generative dataset.
#
# **Input:** `data/train.csv` and `data/test.csv` created by Part 2.
# **Output:** Baseline metrics, predictions, and plots.

# ## 1. Setup and load the Part 2 split

from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = configure_project(__file__)
OUTPUT_DIR = PROJECT_ROOT
STORAGE_MODE = "Local IDE project folder"
DATA_DIR = OUTPUT_DIR / "data"
PLOTS_DIR = OUTPUT_DIR / "plots"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
print("Storage mode:", STORAGE_MODE)
print("Project folder:", OUTPUT_DIR)
print("Dataset folder:", DATA_DIR)
print("Place LLCP2025.XPT in this data folder before running Part 2.")

print("Project folder:", OUTPUT_DIR.resolve())

# ### Persistent Drive handoff
# This notebook reads inputs from and saves outputs to `MyDrive/DP_Medical_Synthetic_Project_2025/`. Run Part 2 first; its CSV outputs remain available to this notebook after runtime resets.

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError("Run Part 2 first. Expected train.csv and test.csv in " + str(DATA_DIR))
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

# ## 2. Create the binary downstream task

TARGET_COL = "DIABETE4"
FEATURE_COLS = [
    "SEXVAR", "GENHLTH", "PHYSHLTH", "MENTHLTH", "EXERANY2", "BPHIGH6",
    "TOLDHI3", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", "CHCCOPD3",
    "ADDEPEV3", "CHCKDNY2", "HAVARTH4", "MARITAL", "EDUCA", "EMPLOY1",
    "INCOME3", "_AGE_G", "_BMI5CAT", "_SMOKER3", "_TOTINDA"
]

def prepare_binary(frame):
    y_code = pd.to_numeric(frame[TARGET_COL], errors="coerce")
    keep = y_code.isin([1, 3])
    X = frame.loc[keep, FEATURE_COLS].copy()
    # Treat BRFSS codes as categories, not continuous measurements.
    X = X.astype("string").fillna("__MISSING__")
    y = (y_code.loc[keep] == 1).astype(int)
    return X, y

X_train, y_train = prepare_binary(train_df)
X_test, y_test = prepare_binary(test_df)
if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError("Both yes/no classes must occur in train and test. Increase MAX_ROWS or review the split.")
print("Binary train rows:", len(X_train), "| test rows:", len(X_test))
print("Train positive rate:", round(y_train.mean(), 4), "| test positive rate:", round(y_test.mean(), 4))

# ## 3. Build preprocessing and baseline classifiers
#
# One-hot encoding is fit only on the training split. Logistic Regression and Random Forest are evaluated on the held-out real test respondents.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# Compatibility across older/newer scikit-learn versions.
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
preprocessor = ColumnTransformer(
    [("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), FEATURE_COLS)],
    remainder="drop"
)
models = {
    "logistic_regression": LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear", random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
}

fitted_models = {}
results = []
prediction_frames = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    print("Training", name)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    results.append(row)
    fitted_models[name] = pipe
    prediction_frames.append(pd.DataFrame({"model": name, "actual": y_test.to_numpy(), "predicted": pred, "probability_yes": proba}))
    print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes yes"], zero_division=0))

baseline_results = pd.DataFrame(results).sort_values("f1", ascending=False)
display(baseline_results)
baseline_results.to_csv(OUTPUT_DIR / "baseline_ml_results.csv", index=False)
pd.concat(prediction_frames, ignore_index=True).to_csv(OUTPUT_DIR / "baseline_ml_predictions.csv", index=False)

# ## 4. Visualize baseline results

metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
ax = baseline_results.set_index("model")[metric_cols].T.plot(kind="bar", figsize=(11, 5))
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Real-train / real-test baseline metrics")
ax.tick_params(axis="x", rotation=0)
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "baseline_ml_metrics.png", dpi=160)
plt.show()

for name, pipe in fitted_models.items():
    pred = pipe.predict(X_test)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False)
    ax.set_title(name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"baseline_cm_{name}.png", dpi=160)
    plt.show()

# ## 5. Save baseline summary for later comparison

summary = {
    "dataset": "CDC 2025 BRFSS",
    "task": "DIABETE4 code 1 (yes) vs code 3 (no)",
    "features": FEATURE_COLS,
    "train_rows_binary": int(len(X_train)),
    "test_rows_binary": int(len(X_test)),
    "best_model_by_f1": str(baseline_results.iloc[0]["model"]),
    "best_f1": float(baseline_results.iloc[0]["f1"]),
}
(OUTPUT_DIR / "baseline_ml_summary.json").write_text(json.dumps(summary, indent=2))
print("Part 4 complete. Results saved in:", OUTPUT_DIR.resolve())
print(summary)

# ## Interpretation note
#
# These are **TRTR** baseline results (train on real data, test on held-out real data). Later, synthetic generators will be assessed using **TSTR** (train on synthetic data, test on the same held-out real data). Compare models using F1, ROC-AUC and balanced accuracy, not accuracy alone, because diabetes classes are imbalanced. These metrics are predictive utility measures, not evidence of clinical validity.
