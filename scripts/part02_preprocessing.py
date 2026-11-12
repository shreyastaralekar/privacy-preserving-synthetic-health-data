#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part2_DP_Medical_Preprocessing.ipynb.

Run from the project root with: python scripts/part02_preprocessing.py
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
print("Part 02 – Part2_DP_Medical_Preprocessing")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 2 – Data Acquisition & Preprocessing
#
# **Goal:** Load the manually uploaded CDC 2025 BRFSS `.XPT` file, select the fixed health-feature schema, create a reproducible sample and train/test split, and save reusable CSV files for Parts 3 and 4.
#
# **Input:** `OUTPUT_DIR/data/LLCP2025.XPT` (or one `.xpt` file in the same folder).
#
# **Output:** `selected_data.csv`, `train.csv`, `test.csv`, and preprocessing metadata.

# ## 1. Setup and reproducibility

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

# ### Dataset location (persistent)
# Upload or copy the CDC 2025 SAS transport file `LLCP2025.XPT` into the `data` folder printed above. Because this folder is on Google Drive, the dataset remains available after a Colab runtime reset. Parts 3 and 4 read the processed CSV files from this same Drive project folder.

# ## 2. Define the project feature schema
#
# The 22 predictor fields and diabetes response are preselected from the CDC 2025 BRFSS file layout. Identifier fields are intentionally excluded.

FEATURE_DOMAINS = {
    "SEXVAR": [1, 2],
    "GENHLTH": [1, 2, 3, 4, 5, 7, 9],
    "PHYSHLTH": list(range(1, 31)) + [77, 88, 99],
    "MENTHLTH": list(range(1, 31)) + [77, 88, 99],
    "EXERANY2": [1, 2, 7, 9],
    "BPHIGH6": [1, 2, 3, 4, 7, 9],
    "TOLDHI3": [1, 2, 7, 9],
    "CVDINFR4": [1, 2, 7, 9],
    "CVDCRHD4": [1, 2, 7, 9],
    "CVDSTRK3": [1, 2, 7, 9],
    "ASTHMA3": [1, 2, 7, 9],
    "CHCCOPD3": [1, 2, 7, 9],
    "ADDEPEV3": [1, 2, 7, 9],
    "CHCKDNY2": [1, 2, 7, 9],
    "HAVARTH4": [1, 2, 7, 9],
    "MARITAL": [1, 2, 3, 4, 5, 6, 9],
    "EDUCA": [1, 2, 3, 4, 5, 6, 9],
    "EMPLOY1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "INCOME3": list(range(1, 12)) + [77, 99],
    "_AGE_G": [1, 2, 3, 4, 5, 6],
    "_BMI5CAT": [1, 2, 3, 4],
    "_SMOKER3": [1, 2, 3, 4],
    "_TOTINDA": [1, 2],
}
TARGET_COL = "DIABETE4"
FEATURE_COLS = list(FEATURE_DOMAINS.keys())
ALL_COLS = FEATURE_COLS + [TARGET_COL]

# ## 3. Load the manually uploaded XPT file

xpt_files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".xpt", ".xport"}]
if not xpt_files:
    raise FileNotFoundError(f"No XPT file found in {DATA_DIR}. Upload the CDC 2025 LLCP XPT file there first.")
if len(xpt_files) > 1:
    raise RuntimeError(f"Multiple XPT files found: {[p.name for p in xpt_files]}. Keep only the 2025 file.")
XPT_PATH = xpt_files[0]
raw = pd.read_sas(XPT_PATH, format="xport", encoding="utf-8")
raw.columns = [str(c).upper() for c in raw.columns]
print("XPT:", XPT_PATH.name, "| Raw shape:", raw.shape)
missing_cols = [c for c in ALL_COLS if c not in raw.columns]
if missing_cols:
    raise KeyError(f"Expected CDC 2025 variables are absent: {missing_cols}. Check the 2025 codebook/variable layout.")

# ## 4. Select, sample and split
#
# The row cap is for free-runtime convenience. Sampling is random and independent of the diabetes response. The split is performed before any model fitting.

MAX_ROWS = 30_000  # Change to None to use all rows if your Colab runtime has sufficient RAM/time.
TEST_SIZE = 0.20

data = raw.loc[:, ALL_COLS].copy()
source_rows = len(data)
if MAX_ROWS is not None and len(data) > MAX_ROWS:
    data = data.sample(n=MAX_ROWS, random_state=SEED)
data = data.reset_index(drop=True)

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(data, test_size=TEST_SIZE, random_state=SEED)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

selected_path = DATA_DIR / "selected_data.csv"
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
data.to_csv(selected_path, index=False)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

quality = pd.DataFrame({
    "missing_rate": data[ALL_COLS].isna().mean(),
    "unique_values": data[ALL_COLS].nunique(dropna=True),
    "non_missing_count": data[ALL_COLS].notna().sum(),
}).sort_values("missing_rate", ascending=False)
quality.to_csv(OUTPUT_DIR / "preprocessing_quality.csv")
metadata = {
    "dataset": "CDC 2025 BRFSS",
    "source_file": XPT_PATH.name,
    "source_rows": int(source_rows),
    "rows_used": int(len(data)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "seed": SEED,
    "test_size": TEST_SIZE,
    "feature_columns": FEATURE_COLS,
    "target_column": TARGET_COL,
    "privacy_unit": "one BRFSS respondent row",
}
(OUTPUT_DIR / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2))
print("Saved:", selected_path, train_path, test_path)
print("Train:", train_df.shape, "Test:", test_df.shape)
display(quality)
display(train_df.head())

# ## 5. Check diabetes response codes
#
# BRFSS `DIABETE4` is retained with all response categories for generative modelling. Only clear yes/no answers (1 and 3) will be used for the binary downstream task in Part 4.

print("DIABETE4 distribution (all selected respondents):")
display(data[TARGET_COL].value_counts(dropna=False).sort_index().rename("count").to_frame())
print("\nPreprocessing complete. Parts 3 and 4 can now load the saved CSV files.")

# ## Output files
#
# All files are saved under `dp_medical_synthetic_2025/`. Keep this folder for the next parts. In Colab, download it or keep it in the same runtime/Google Drive between sessions.
