#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part3_DP_Medical_EDA.ipynb.

Run from the project root with: python scripts/part03_eda.py
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
print("Part 03 – Part3_DP_Medical_EDA")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 3 – Exploratory Data Analysis (EDA)
#
# **Goal:** Summarize the prepared CDC 2025 BRFSS sample, check missingness and response distributions, and export report-ready plots.
#
# **Input:** `data/selected_data.csv` created by Part 2.
# **Output:** EDA tables and figures in `plots/`.

# ## 1. Setup and load prepared data

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

DATA_PATH = DATA_DIR / "selected_data.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError("Run Part 2 first. Expected file: " + str(DATA_PATH))
data = pd.read_csv(DATA_PATH)
data.columns = [str(c).upper() for c in data.columns]
print("Prepared data shape:", data.shape)
display(data.head())

# ## 2. Data quality and missing values

quality = pd.DataFrame({
    "missing_count": data.isna().sum(),
    "missing_rate": data.isna().mean(),
    "unique_values": data.nunique(dropna=True),
}).sort_values("missing_rate", ascending=False)
display(quality)
quality.to_csv(OUTPUT_DIR / "eda_data_quality.csv")

plt.figure(figsize=(10, 6))
quality.head(15).sort_values("missing_rate")["missing_rate"].plot(kind="barh")
plt.xlabel("Missing-value proportion")
plt.title("Top 15 variables by missingness")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eda_missingness.png", dpi=160)
plt.show()

# ## 3. Diabetes response distribution

target_counts = data["DIABETE4"].value_counts(dropna=False).sort_index()
print("DIABETE4 code counts:")
display(target_counts.rename("count").to_frame())
labels = {1: "Diabetes yes", 2: "Yes – pregnancy", 3: "No", 4: "Prediabetes", 7: "Don't know", 9: "Refused"}
plot_counts = target_counts.copy()
plot_counts.index = [labels.get(int(x), str(x)) if pd.notna(x) else "Missing" for x in plot_counts.index]
plt.figure(figsize=(9, 4.5))
plot_counts.plot(kind="bar")
plt.ylabel("Respondents")
plt.title("BRFSS diabetes response distribution (DIABETE4)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eda_diabetes_response.png", dpi=160)
plt.show()

# ## 4. Key categorical feature distributions

key_cols = ["SEXVAR", "GENHLTH", "EXERANY2", "BPHIGH6", "_AGE_G", "_BMI5CAT", "_SMOKER3", "_TOTINDA"]
fig, axes = plt.subplots(4, 2, figsize=(13, 16))
for ax, col in zip(axes.flat, key_cols):
    vals = data[col].value_counts(dropna=False).sort_index()
    vals.index = ["Missing" if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v) for v in vals.index]
    vals.plot(kind="bar", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("BRFSS response code")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eda_key_feature_distributions.png", dpi=160)
plt.show()

# ## 5. Health-days variables (valid numeric codes only)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["PHYSHLTH", "MENTHLTH"]):
    vals = pd.to_numeric(data[col], errors="coerce")
    vals = vals[vals.between(1, 30)]
    ax.hist(vals, bins=30)
    ax.set_title(f"{col}: valid reported days (1–30)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Respondents")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eda_health_days.png", dpi=160)
plt.show()

# ## 6. Simple association snapshot
#
# These response codes are categorical/ordinal; do not interpret Pearson correlation on arbitrary codes as a clinical relationship. This section reports Cramér's V for a small set of categorical variables.

def cramers_v(x, y):
    table = pd.crosstab(x.fillna("MISSING").astype(str), y.fillna("MISSING").astype(str))
    n = table.to_numpy().sum()
    if n == 0 or min(table.shape) < 2:
        return np.nan
    chi2 = __import__("scipy.stats", fromlist=["chi2_contingency"]).chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    r, k = table.shape
    # Bias-corrected Cramer's V
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) if n > 1 else 0
    rc = r - ((r-1)**2)/(n-1) if n > 1 else r
    kc = k - ((k-1)**2)/(n-1) if n > 1 else k
    denom = min((kc-1), (rc-1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

assoc_cols = ["DIABETE4", "GENHLTH", "EXERANY2", "BPHIGH6", "_AGE_G", "_BMI5CAT", "_SMOKER3"]
assoc = pd.DataFrame(index=assoc_cols, columns=assoc_cols, dtype=float)
for a in assoc_cols:
    for b in assoc_cols:
        assoc.loc[a, b] = 1.0 if a == b else cramers_v(data[a], data[b])
display(assoc.round(3))
plt.figure(figsize=(8, 6))
sns.heatmap(assoc, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis")
plt.title("Categorical association snapshot (Cramér's V)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eda_cramers_v.png", dpi=160)
plt.show()
assoc.to_csv(OUTPUT_DIR / "eda_cramers_v.csv")

# ## 7. Export EDA summary

summary = {
    "rows": int(len(data)),
    "columns": int(data.shape[1]),
    "overall_missing_rate": float(data.isna().mean().mean()),
    "diabetes_yes_code_1_rate": float((pd.to_numeric(data["DIABETE4"], errors="coerce") == 1).mean()),
    "diabetes_no_code_3_rate": float((pd.to_numeric(data["DIABETE4"], errors="coerce") == 3).mean()),
}
(OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
print("EDA complete. Tables and plots saved to:", OUTPUT_DIR.resolve())
print(summary)
