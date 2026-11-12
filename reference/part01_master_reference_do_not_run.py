#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part1_DP_Medical_Project.ipynb.

Run from the project root with: python scripts/...
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
print("Part 01 – Part1_DP_Medical_Project")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# REFERENCE ONLY: This is the original all-in-one notebook conversion.
# Do not run it together with Parts 2-14; it duplicates the modular pipeline.

# # Part 1 – Privacy-Preserving Synthetic Medical Data Project
#
# **Project:** Differentially Private Diffusion for Tabular Health Data  
# **Dataset:** CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS)  
# **Environment:** Google Colab  
# **Model:** Tabular DDPM + Opacus DP-SGD
#
# This is the main end-to-end research notebook. Upload the CDC `.XPT` file manually to the `data` folder before running. The later parts (2–4) provide shorter notebooks for data preparation, EDA, and baseline ML.
#
# **Persistent storage:** Google Drive is mounted in the setup cell. Place the manually downloaded `LLCP2025.XPT` file in `MyDrive/DP_Medical_Synthetic_Project_2025/data/`. All outputs, plots, checkpoints and generated datasets are written to this same Drive folder and survive Colab runtime resets.

# ## 1. Research questions, scope, and hypotheses
#
# ### Research questions
#
# - **RQ1 – Fidelity:** How well does the diffusion model reproduce marginal categorical distributions and pairwise feature associations?
# - **RQ2 – Utility:** Can a classifier trained only on synthetic data (TSTR) retain predictive utility when evaluated on held-out real respondents?
# - **RQ3 – Privacy:** How do the formal privacy budgets and empirical membership-inference / nearest-neighbour audits change across non-private and DP training?
# - **RQ4 – Trade-off:** How does utility change as the target privacy budget \(\\epsilon\\) changes?
#
# ### Hypotheses (to be tested, not assumed)
#
# - Stronger privacy (smaller epsilon) may reduce distribution fidelity and TSTR utility.
# - Increasing epsilon may improve utility, but it weakens the formal privacy bound.
# - Empirical attacks may show different behaviour from the formal guarantee; an attack AUC near 0.5 is evidence only about the tested attack, not a proof of privacy.
#
# ### Scope and terminology
#
# - One row represents one BRFSS survey respondent for the 2025 annual cross-sectional file. We therefore define the privacy unit as **one respondent record**.
# - Only a fixed set of core health and demographic variables is used. Direct/internal identifiers, state, survey sequence numbers, and survey weights are excluded from model inputs.
# - Numeric-looking BRFSS codes are treated as categorical codes in this prototype. One-hot encoding is fixed from the public questionnaire/codebook schema; missing and out-of-domain values receive explicit sentinel categories.
# - The generator is a lightweight Gaussian DDPM baseline for mixed tabular categorical records represented in one-hot space. It is **inspired by** tabular diffusion literature, not a reproduction of the full TabDDPM or TabDiff architecture.

# ## 2. Method and privacy design
#
# ### Pipeline
#
# 1. Download CDC 2025 BRFSS SAS transport data and read the XPT file.
# 2. Keep all respondent rows and all `DIABETE4` response categories for generator training; restrict to unambiguous yes/no responses (codes 1 and 3) only for the downstream binary classifier.
# 3. Use public, fixed category domains from the BRFSS questionnaire/codebook. No learned scaler or private-data-fitted category vocabulary is required.
# 4. Split respondents into train/test partitions before model fitting.
# 5. Train one non-private DDPM and three DP-DDPMs using Opacus `PrivacyEngine.make_private_with_epsilon`.
# 6. Generate synthetic rows, decode each one-hot feature block using argmax, and evaluate fidelity, TSTR utility, and empirical privacy risks.
#
# ### DDPM objective
#
# The forward process adds Gaussian noise:
#
# \[
#  x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\epsilon,\quad \\epsilon\\sim\\mathcal{N}(0,I)
# \]
#
# The denoiser learns to predict the added noise:
#
# \[
#  L_{\\mathrm{DDPM}} = \\mathbb{E}_{x_0,t,\\epsilon}\\left[\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|_2^2\\right]
# \]
#
# ### DP-SGD mechanism
#
# For each respondent's per-example gradient \(g_i\), Opacus clips its norm to \(C\), aggregates clipped gradients, and adds calibrated Gaussian noise before the optimizer update. The Opacus accountant tracks composition over all private optimizer steps and returns the achieved epsilon for the selected delta.
#
# **Formal guarantee requirements:** use the private model, private optimizer, and private data loader returned by Opacus for every training step; do not bypass the DP optimizer; do not train further on private rows after the accounting run; and do not release unaccounted private-data-derived preprocessing or tuning outputs. The data pipeline below uses fixed public code domains and record-local transforms. Empirical metrics are audits, not substitutes for the DP accountant.
#
# **Delta:** the notebook chooses \(\\delta = \\min(10^{-5}, 1/(10N))\), where \(N\) is the number of training respondents, ensuring it is smaller than \(1/N\) for this run.
#
# **Randomness note:** `secure_mode=False` is used for compatibility and speed in this free Colab research notebook. Opacus documents that `secure_mode=True` uses a cryptographically strong random-number generator and requires `torchcsprng`; the default pseudo-random mode is not a production-grade cryptographic implementation. Therefore, treat the reported accountant epsilon as the theoretical Gaussian-mechanism accounting result for this research run, not as a deployment certification. For a high-assurance deployment, install a compatible secure RNG, set `secure_mode=True`, rerun the complete experiment, and obtain a dedicated privacy review.

# ## 3. Data source and related methods
#
# - **CDC 2025 BRFSS annual survey data and documentation:** https://www.cdc.gov/brfss/annual_data/annual_2025.html
# - **CDC 2025 variable layout:** https://www.cdc.gov/brfss/annual_data/2025/llcp_varlayout_25_onecolumn.html
# - **CDC BRFSS codebook / questionnaire documentation:** linked from the CDC 2025 annual data page.
# - **TabDDPM (ICML 2023), paper and official implementation:** https://arxiv.org/abs/2209.15421 and https://github.com/yandex-research/tab-ddpm
# - **TabDiff (ICLR 2025), mixed-type diffusion research:** https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html
# - **Opacus documentation:** https://opacus.ai/api/privacy_engine.html
# - **DP-SGD background:** Abadi et al., “Deep Learning with Differential Privacy” (2016), https://arxiv.org/abs/1607.00133
#
# **Dataset notes:** CDC reports 356,158 records for the 2025 annual file, released August 2026, and 284 variables in the SAS transport file. The annual file includes the diabetes variable `DIABETE4`. BRFSS has a complex survey design; this notebook does not apply survey weights in generative training, so the generated distribution is an unweighted sample distribution and should not be described as a nationally weighted prevalence estimate.

# ## 4. Colab runtime setup
#
# In Colab, select **Runtime → Change runtime type → T4 GPU** if available. CPU mode is supported for a reduced run. The free GPU pool is not guaranteed, so the configuration below uses a compact MLP and a configurable respondent sample.

# Install only the additional libraries; keep the Colab-provided PyTorch build.
# [Notebook install command moved to requirements.txt]

import os, sys, json, math, time, random, zipfile, warnings, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tqdm.auto import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.neighbors import NearestNeighbors

import opacus
from opacus import PrivacyEngine

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Opacus:", opacus.__version__)
print("Pandas:", pd.__version__)
print("CUDA available:", torch.cuda.is_available())

# ## 5. Reproducibility and experiment configuration
#
# The defaults are deliberately modest for free Colab. Set `MAX_ROWS = None` to train on every respondent row, but first verify available RAM/GPU and runtime budget. Increase `EPOCHS` or `TIMESTEPS` only after the complete small run works.

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Colab-friendly defaults
MAX_ROWS = 30_000          # set to None to use all 2025 respondents
TEST_SIZE = 0.20
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 2e-4
TIMESTEPS = 100
HIDDEN_DIM = 128
TIME_EMBED_DIM = 32
MAX_GRAD_NORM = 1.0
SECURE_MODE = False  # requires a compatible torchcsprng install when set True
TARGET_EPSILONS = [1.0, 3.0, 8.0]

# Audit compute caps (audits are empirical approximations, not formal guarantees)
MIA_MAX_PER_GROUP = 4_000
DCR_REFERENCE_N = 1_500
DCR_SYNTH_N = 500
SYNTHETIC_CLASSIFIER_N = 4_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Persistent Google Drive project storage (shared by Parts 1–4)
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive')
    if not DRIVE_ROOT.exists():
        raise RuntimeError("Google Drive did not mount. Check the authorization step and rerun this cell.")
    OUTPUT_DIR = configure_project(__file__)
    STORAGE_MODE = "Google Drive (persistent)"
except ImportError:
    # Local Jupyter / VS Code fallback. This is local storage, not Google Drive.
    OUTPUT_DIR = configure_project(__file__)
    STORAGE_MODE = "Local project folder"

DATA_DIR = OUTPUT_DIR / "data"
PLOTS_DIR = OUTPUT_DIR / "plots"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for folder in (OUTPUT_DIR, DATA_DIR, PLOTS_DIR, CHECKPOINT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

print("Storage mode:", STORAGE_MODE)
print("Persistent project folder:", OUTPUT_DIR)
print("Dataset folder:", DATA_DIR)
print("IMPORTANT: Put LLCP2025.XPT in this Drive data folder before running preprocessing.")


print("Device:", DEVICE)
print("Output directory:", OUTPUT_DIR.resolve())
print("Configuration:", {k: v for k, v in {
    "MAX_ROWS": MAX_ROWS, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
    "TIMESTEPS": TIMESTEPS, "TARGET_EPSILONS": TARGET_EPSILONS
}.items()})

# ## 6. Load the manually uploaded CDC 2025 dataset
#
# Place the uncompressed CDC SAS transport file (`.XPT`) in `OUTPUT_DIR/data/`. This notebook does not download or extract a ZIP file. It validates the file and loads it in the next cell.

CDC_PAGE = "https://www.cdc.gov/brfss/annual_data/annual_2025.html"
CDC_XPT_ZIP_URL = "https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip"  # source reference only
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / "LLCP2025XPT.zip"  # retained only for the results-bundle exclusion check

# The file must be uploaded/extracted manually into the data folder.
preferred_path = DATA_DIR / "LLCP2025.XPT"
if preferred_path.exists():
    XPT_PATH = preferred_path
else:
    xpt_files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".xpt", ".xport"}]
    if len(xpt_files) == 1:
        XPT_PATH = xpt_files[0]
    elif len(xpt_files) == 0:
        raise FileNotFoundError(
            f"No .XPT file found in {DATA_DIR}. Upload the CDC 2025 LLCP XPT file into this folder."
        )
    else:
        raise RuntimeError(f"Multiple XPT files found in {DATA_DIR}: {[p.name for p in xpt_files]}. Keep only the 2025 file.")

if XPT_PATH.stat().st_size == 0:
    raise RuntimeError(f"XPT file is empty: {XPT_PATH}")

print("DATASET READY")
print("XPT file:", XPT_PATH)
print("File size:", round(XPT_PATH.stat().st_size / (1024**2), 2), "MB")

# ## 7. Load and inspect raw 2025 BRFSS data
#
# The annual file contains many variables; the model will use a prespecified subset of core variables. We keep the raw data only for this session and do not send it to any third-party API.

raw = pd.read_sas(XPT_PATH, format="xport", encoding="utf-8")
raw.columns = [str(c).upper() for c in raw.columns]
print("Raw shape:", raw.shape)
print("First columns:", raw.columns[:25].tolist())
display(raw.head(3))

# ## 8. Define fixed public feature domains
#
# The following core BRFSS variables are listed in the 2025 CDC variable layout. They cover demographics, general/physical/mental health, activity, chronic conditions, and socioeconomic factors. Values are retained as category codes. The domains below are specified from the BRFSS coding conventions; `__MISSING__` and `__OTHER__` are fixed sentinel categories, not categories learned from the training sample.
#
# For a final academic report, cross-check every code label against the CDC 2025 codebook linked on the annual data page. The model uses the codes rather than interpreting them as continuous measurements. `DIABETE4` retains codes 1, 2, 3, 4, 7 and 9 plus missing/other sentinels for generative training.

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
TARGET_DOMAIN = ["1", "2", "3", "4", "7", "9"]  # BRFSS DIABETE4 response codes; missing/other sentinels are added below
FEATURE_COLS = list(FEATURE_DOMAINS.keys())
ALL_COLS = FEATURE_COLS + [TARGET_COL]

missing_cols = [c for c in ALL_COLS if c not in raw.columns]
if missing_cols:
    raise KeyError(
        f"Expected 2025 BRFSS variables are absent: {missing_cols}. "
        "Check the CDC variable layout and update the feature schema before training."
    )

print("Number of model features:", len(FEATURE_COLS))
print("Selected columns:", ALL_COLS)

# ## 9. Prepare the respondent-level train/test split
#
# **Privacy-first choice:** the generator is trained on all 2025 respondent rows, retaining the available `DIABETE4` response categories (including pregnancy-only, prediabetes/borderline, don't-know, refused, missing, and other codes). We do not filter people out based on the target before training. This makes the generative training task cover the full respondent table used in this experiment.
#
# For the downstream *binary diabetes classification* task only, we later evaluate respondents with an unambiguous label: code 1 (yes) or code 3 (no). Codes 2, 4, 7, 9, missing, and other are not forced into either binary class.
#
# A fixed-size random sample and train/test split are performed without stratification, so selection does not depend on the respondent's diabetes label. By default, 30,000 rows are used for free Colab; set `MAX_ROWS = None` to use all records. One BRFSS respondent row is one privacy unit in this single-year file.

data = raw.loc[:, ALL_COLS].copy()
source_rows = len(data)

# Random row sampling is independent of response values. It is a compute cap, not class balancing.
if MAX_ROWS is not None and source_rows > MAX_ROWS:
    data = data.sample(n=int(MAX_ROWS), random_state=SEED)
data = data.reset_index(drop=True)

# Split without stratification so the assignment does not depend on the target label.
train_df, test_df = train_test_split(data, test_size=TEST_SIZE, random_state=SEED)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Rows in downloaded 2025 file: {source_rows:,}")
print(f"Rows used after optional random cap: {len(data):,}")
print(f"Train respondents: {len(train_df):,}; test respondents: {len(test_df):,}")
print("DIABETE4 code distribution in training data:")
display(train_df[TARGET_COL].value_counts(dropna=False, normalize=True).rename("proportion").to_frame())

# Data-quality summary for the selected variables; reported, not used to fit preprocessing.
quality = pd.DataFrame({
    "missing_rate": train_df[ALL_COLS].isna().mean(),
    "n_observed_codes": train_df[ALL_COLS].nunique(dropna=True),
}).sort_values("missing_rate", ascending=False)
display(quality)

# ## 10. Fixed-schema one-hot encoding and decoding
#
# The encoder does not fit categories from the respondent sample. Each column has a public, predefined domain plus two fixed sentinel categories. This keeps preprocessing reproducible and avoids a private-data-derived category vocabulary. The one-hot representation is the continuous state space used by this lightweight Gaussian DDPM. During decoding, argmax is applied independently to each feature block, guaranteeing one valid code per feature block.

MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"
DOMAINS = {
    col: [str(v) for v in FEATURE_DOMAINS[col]] + [MISSING_TOKEN, OTHER_TOKEN]
    for col in FEATURE_COLS
}
DOMAINS[TARGET_COL] = TARGET_DOMAIN + [MISSING_TOKEN, OTHER_TOKEN]

CATEGORY_TO_INDEX = {
    col: {value: idx for idx, value in enumerate(DOMAINS[col])}
    for col in ALL_COLS
}


def normalize_code(value, column):
    """Map a BRFSS scalar to a fixed public-schema category string."""
    # Preserve sentinel strings after decoding synthetic rows.
    if isinstance(value, str) and value in CATEGORY_TO_INDEX[column]:
        return value
    if pd.isna(value):
        return MISSING_TOKEN
    try:
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            return OTHER_TOKEN
        code = str(int(numeric))
    except (TypeError, ValueError):
        return OTHER_TOKEN
    if code in CATEGORY_TO_INDEX[column]:
        return code
    return OTHER_TOKEN if column != TARGET_COL else None


def encode_frame(frame: pd.DataFrame, columns=ALL_COLS) -> np.ndarray:
    blocks = []
    for col in columns:
        domain = DOMAINS[col]
        mapping = CATEGORY_TO_INDEX[col]
        values = frame[col].map(lambda v: normalize_code(v, col)).tolist()
        if any(v is None for v in values):
            raise ValueError(f"Unexpected target code found in {col}.")
        indices = np.fromiter((mapping[v] for v in values), dtype=np.int64, count=len(values))
        blocks.append(np.eye(len(domain), dtype=np.float32)[indices])
    return np.concatenate(blocks, axis=1).astype(np.float32)


def decode_matrix(matrix: np.ndarray, columns=ALL_COLS) -> pd.DataFrame:
    matrix = np.asarray(matrix)
    decoded = {}
    start = 0
    for col in columns:
        domain = DOMAINS[col]
        stop = start + len(domain)
        block = matrix[:, start:stop]
        indices = np.argmax(block, axis=1)
        decoded[col] = [domain[int(i)] for i in indices]
        start = stop
    if start != matrix.shape[1]:
        raise ValueError("Encoded dimension does not match the declared category schema.")
    return pd.DataFrame(decoded)


X_train_tab = encode_frame(train_df, ALL_COLS)
X_test_tab = encode_frame(test_df, ALL_COLS)
INPUT_DIM = X_train_tab.shape[1]

feature_slices = {}
_cursor = 0
for _col in ALL_COLS:
    feature_slices[_col] = (_cursor, _cursor + len(DOMAINS[_col]))
    _cursor += len(DOMAINS[_col])

print("Encoded train shape:", X_train_tab.shape)
print("Encoded test shape:", X_test_tab.shape)
print("Input dimension:", INPUT_DIM)
assert np.allclose(X_train_tab.sum(axis=1).reshape(-1, 1).min(), len(ALL_COLS))

# ## 11. Baseline downstream ML: train on real, test on real (TRTR)
#
# A logistic-regression classifier is used as a consistent downstream evaluator. We report accuracy, precision, recall, F1, and ROC-AUC. Because diabetes labels are imbalanced, macro F1 and ROC-AUC are especially useful. The classifier is an evaluation model, separate from the DP generative model.

def binary_eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only clear yes/no diabetes responses for the downstream binary task."""
    mask = frame[TARGET_COL].map(lambda v: normalize_code(v, TARGET_COL)).isin(["1", "3"])
    return frame.loc[mask].copy().reset_index(drop=True)


def binary_labels(frame: pd.DataFrame) -> np.ndarray:
    # BRFSS DIABETE4: code 1 = yes; code 3 = no. Other responses are excluded by binary_eligible().
    mapped = frame[TARGET_COL].map(lambda v: normalize_code(v, TARGET_COL)).map({"1": 1, "3": 0})
    if mapped.isna().any():
        raise ValueError("binary_labels received a response outside DIABETE4 codes 1 and 3.")
    return mapped.to_numpy(dtype=np.int64)


def classification_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
    }


def train_evaluate_tstr(train_records: pd.DataFrame, test_records: pd.DataFrame, label: str):
    X_train = encode_frame(train_records, FEATURE_COLS)
    X_test = encode_frame(test_records, FEATURE_COLS)
    y_train = binary_labels(train_records)
    y_test = binary_labels(test_records)
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{label}: synthetic training data contains only one target class.")
    clf = LogisticRegression(
        max_iter=300, class_weight="balanced", solver="lbfgs", random_state=SEED
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    return {"model": label, **classification_metrics(y_test, pred, prob)}, clf


binary_train_real = binary_eligible(train_df)
binary_test_real = binary_eligible(test_df)
if binary_train_real[TARGET_COL].nunique() < 2 or binary_test_real[TARGET_COL].nunique() < 2:
    raise ValueError("Both DIABETE4 yes/no classes must appear in train and test for downstream evaluation.")
print(f"Binary task respondents: train={len(binary_train_real):,}; test={len(binary_test_real):,}")
trtr_metrics, trtr_classifier = train_evaluate_tstr(binary_train_real, binary_test_real, "TRTR_real_train")
display(pd.DataFrame([trtr_metrics]).set_index("model"))

# ## 12. Define the time-conditioned diffusion model
#
# The denoiser is a compact MLP with a sinusoidal timestep embedding. It contains only Linear and SiLU layers (no BatchNorm), which keeps per-example gradient computation practical with Opacus. All feature blocks are concatenated into one vector; the categorical structure is restored by block-wise argmax after sampling.

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class TabularDenoiser(nn.Module):
    def __init__(self, input_dim: int, time_dim: int = TIME_EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        return self.network(torch.cat([x_t, t_emb], dim=1))


print("Denoiser parameters:", sum(p.numel() for p in TabularDenoiser(INPUT_DIM).parameters()))

# ## 13. Noise schedule and DDPM equations
#
# A cosine cumulative-alpha schedule is used to make the terminal distribution close to Gaussian over a relatively small number of diffusion steps. The sampler uses the standard DDPM posterior mean and variance. The chosen timestep count is a computational compromise for free Colab.

def cosine_noise_schedule(timesteps: int, device=DEVICE, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "posterior_variance": posterior_variance,
    }


SCHEDULE = cosine_noise_schedule(TIMESTEPS)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule=SCHEDULE):
    alpha_bar_t = schedule["alpha_bars"][t].unsqueeze(1)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def diffusion_per_example_loss(model, x0, schedule=SCHEDULE):
    batch_size = x0.shape[0]
    t = torch.randint(0, TIMESTEPS, (batch_size,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    predicted_noise = model(x_t, t)
    # One scalar per respondent is required for per-example gradient clipping.
    return (predicted_noise - noise).pow(2).mean(dim=1)

# Sanity check schedule
print("alpha_bar at final timestep:", float(SCHEDULE["alpha_bars"][-1].item()))

# ## 14. Training utilities
#
# The non-private baseline uses ordinary Adam. The private models are wrapped by Opacus and then trained with the returned private optimizer and private data loader. The loss is reduced to a mean only after retaining per-example losses, allowing Opacus to compute per-record gradients.

def make_tensor_loader(encoded_array: np.ndarray, batch_size=BATCH_SIZE, shuffle=True):
    tensor = torch.from_numpy(np.asarray(encoded_array, dtype=np.float32))
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def train_nonprivate(encoded_array: np.ndarray, epochs=EPOCHS):
    model = TabularDenoiser(INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = make_tensor_loader(encoded_array)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for (x0,) in loader:
            x0 = x0.to(DEVICE)
            per_example = diffusion_per_example_loss(model, x0)
            loss = per_example.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else np.nan)
        print(f"Non-private | epoch {epoch+1:02d}/{epochs} | loss={history[-1]:.5f}")
    return model, history


def train_private(encoded_array: np.ndarray, target_epsilon: float, delta: float, epochs=EPOCHS):
    model = TabularDenoiser(INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = make_tensor_loader(encoded_array)
    privacy_engine = PrivacyEngine(accountant="prv", secure_mode=SECURE_MODE)

    # Opacus selects a noise multiplier to target epsilon at the requested delta/epochs.
    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=float(target_epsilon),
        target_delta=float(delta),
        epochs=int(epochs),
        max_grad_norm=float(MAX_GRAD_NORM),
        poisson_sampling=True,
        rand_on_empty=True,
    )

    history = []
    for epoch in range(epochs):
        private_model.train()
        losses = []
        for (x0,) in private_loader:
            x0 = x0.to(DEVICE)
            if x0.shape[0] == 0:
                continue
            per_example = diffusion_per_example_loss(private_model, x0)
            loss = per_example.mean()
            private_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            private_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else np.nan)
        print(f"DP epsilon target={target_epsilon:g} | epoch {epoch+1:02d}/{epochs} | loss={history[-1]:.5f}")

    achieved_epsilon = float(privacy_engine.get_epsilon(delta=float(delta)))
    actual_noise_multiplier = float(getattr(private_optimizer, "noise_multiplier", np.nan))
    print(f"Accountant: achieved epsilon={achieved_epsilon:.4f}, delta={delta:.3e}, noise={actual_noise_multiplier:.4f}")
    return private_model, history, privacy_engine, achieved_epsilon, actual_noise_multiplier

# ## 15. Train the non-private and DP diffusion models
#
# The notebook trains one baseline plus one private model for each target epsilon. The accountant's **achieved** epsilon is the value to report; the target epsilon is only the requested budget. Keep the runtime alive until checkpoints and outputs are saved.
#
# If a free Colab session runs out of resources, lower `MAX_ROWS`, `EPOCHS`, or `BATCH_SIZE` and rerun from the configuration cell. Do not present a target epsilon as an achieved epsilon.

# Calculate delta from the actual respondent-level training set size.
N_TRAIN = len(train_df)
DELTA = min(1e-5, 1.0 / (10.0 * N_TRAIN))
print(f"Training respondents N={N_TRAIN:,}; delta={DELTA:.3e}; 1/N={1/N_TRAIN:.3e}")
assert DELTA < 1.0 / N_TRAIN

models = {}
training_histories = {}
privacy_records = []

# Non-private reference model
start_time = time.time()
nonprivate_model, nonprivate_history = train_nonprivate(X_train_tab, epochs=EPOCHS)
models["non_private"] = {"model": nonprivate_model, "target_epsilon": np.inf, "achieved_epsilon": np.inf}
training_histories["non_private"] = nonprivate_history
nonprivate_state = getattr(nonprivate_model, "_module", nonprivate_model)
torch.save(nonprivate_state.state_dict(), OUTPUT_DIR / "checkpoints" / "ddpm_non_private.pt")
print(f"Non-private training time: {(time.time()-start_time)/60:.1f} minutes")

# Differentially private models
for eps in TARGET_EPSILONS:
    start_time = time.time()
    private_model, history, accountant, achieved_eps, noise_mult = train_private(
        X_train_tab, target_epsilon=eps, delta=DELTA, epochs=EPOCHS
    )
    key = f"dp_eps_{eps:g}"
    models[key] = {
        "model": private_model,
        "target_epsilon": float(eps),
        "achieved_epsilon": achieved_eps,
        "delta": DELTA,
        "noise_multiplier": noise_mult,
    }
    training_histories[key] = history
    module_to_save = getattr(private_model, "_module", getattr(private_model, "module", private_model))
    torch.save(module_to_save.state_dict(), OUTPUT_DIR / "checkpoints" / f"{key}.pt")
    privacy_records.append({
        "model": key,
        "target_epsilon": float(eps),
        "achieved_epsilon": achieved_eps,
        "delta": DELTA,
        "noise_multiplier": noise_mult,
        "epochs": EPOCHS,
        "respondent_train_n": N_TRAIN,
        "batch_size_nominal": BATCH_SIZE,
        "max_grad_norm": MAX_GRAD_NORM,
    })
    print(f"{key} training time: {(time.time()-start_time)/60:.1f} minutes")

privacy_df = pd.DataFrame(privacy_records)
display(privacy_df)
privacy_df.to_csv(OUTPUT_DIR / "privacy_accounting.csv", index=False)

# ## 16. Reverse diffusion sampling and categorical decoding
#
# The reverse process starts from Gaussian noise and iteratively predicts a less noisy vector. The generated vector is decoded one feature block at a time. The resulting values are valid *codes* in the declared schema; this does not guarantee every cross-variable combination is clinically plausible.

@torch.no_grad()
def sample_diffusion(model, n_samples: int, input_dim: int, schedule=SCHEDULE, batch_size=512):
    model.eval()
    outputs = []
    betas = schedule["betas"]
    alphas = schedule["alphas"]
    alpha_bars = schedule["alpha_bars"]
    posterior_variance = schedule["posterior_variance"]

    for start in tqdm(range(0, n_samples, batch_size), desc="Reverse diffusion"):
        current_n = min(batch_size, n_samples - start)
        x = torch.randn(current_n, input_dim, device=DEVICE)
        for step in reversed(range(TIMESTEPS)):
            t = torch.full((current_n,), step, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t)
            beta_t = betas[step]
            alpha_t = alphas[step]
            alpha_bar_t = alpha_bars[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            )
            if step > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(torch.clamp(posterior_variance[step], min=1e-20)) * z
            else:
                x = mean
        outputs.append(x.cpu().numpy())
    return np.concatenate(outputs, axis=0)


synthetic_datasets = {}
SYNTHETIC_N = len(train_df)
for model_name, bundle in models.items():
    print("\nGenerating:", model_name)
    generated_matrix = sample_diffusion(bundle["model"], SYNTHETIC_N, INPUT_DIM)
    synthetic_frame = decode_matrix(generated_matrix, ALL_COLS)
    synthetic_datasets[model_name] = synthetic_frame
    filename = OUTPUT_DIR / f"synthetic_{model_name}.csv"
    synthetic_frame.to_csv(filename, index=False)
    print("Saved", filename, "shape", synthetic_frame.shape)

# Short preview; codes are retained to keep exact reproducibility.
display(synthetic_datasets["non_private"].head())

# ## 17. Statistical fidelity metrics
#
# We report:
#
# - **Mean Jensen–Shannon distance (JSD):** compares one-variable category distributions across the fixed code domains; lower is better.
# - **Cramér's V matrix MAE:** compares pairwise categorical association strengths for a selected subset; lower is better.
# - **Synthetic-vs-real classifier AUC:** a classifier tries to distinguish synthetic records from held-out real records; 0.5 is ideal for indistinguishability under this particular attack, while higher values indicate easier separation.
#
# These are practical utility/fidelity metrics, not privacy guarantees.

def category_js_distance(real_df, synth_df, columns=ALL_COLS):
    scores = {}
    for col in columns:
        domain = DOMAINS[col]
        real_vals = real_df[col].map(lambda v: normalize_code(v, col))
        synth_vals = synth_df[col].map(lambda v: normalize_code(v, col))
        p = real_vals.value_counts().reindex(domain, fill_value=0).to_numpy(dtype=float)
        q = synth_vals.value_counts().reindex(domain, fill_value=0).to_numpy(dtype=float)
        p = (p + 1e-8) / (p.sum() + 1e-8 * len(p))
        q = (q + 1e-8) / (q.sum() + 1e-8 * len(q))
        scores[col] = float(jensenshannon(p, q, base=2))
    return scores


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    # Normalize real missing values and synthetic sentinel strings consistently.
    x_name = x.name
    y_name = y.name
    x_values = x.map(lambda v: normalize_code(v, x_name)).astype(str)
    y_values = y.map(lambda v: normalize_code(v, y_name)).astype(str)
    table = pd.crosstab(x_values, y_values)
    if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() == 0:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    # Bergsma bias correction
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denominator)) if denominator > 0 else 0.0


def cramers_v_matrix(frame: pd.DataFrame, columns):
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    for i, col_a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_b = columns[j]
            value = cramers_v(frame[col_a], frame[col_b])
            matrix.loc[col_a, col_b] = value
            matrix.loc[col_b, col_a] = value
    return matrix


ASSOCIATION_COLS = [
    "GENHLTH", "PHYSHLTH", "EXERANY2", "BPHIGH6", "TOLDHI3",
    "CVDINFR4", "CVDSTRK3", "MARITAL", "EDUCA", "INCOME3", "_AGE_G", "_BMI5CAT"
]


def synthetic_real_classifier_auc(real_df, synth_df, max_n=SYNTHETIC_CLASSIFIER_N):
    n = min(max_n, len(real_df), len(synth_df))
    real_part = real_df.sample(n=n, random_state=SEED)
    synth_part = synth_df.sample(n=n, random_state=SEED)
    combined = pd.concat([real_part, synth_part], ignore_index=True)
    labels = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    X = encode_frame(combined, FEATURE_COLS)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED)
    clf.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


fidelity_rows = []
fidelity_details = {}
real_reference = train_df.copy()
real_assoc = cramers_v_matrix(real_reference, ASSOCIATION_COLS)

for name, synth_df in synthetic_datasets.items():
    js_by_col = category_js_distance(real_reference, synth_df)
    synth_assoc = cramers_v_matrix(synth_df, ASSOCIATION_COLS)
    upper = np.triu_indices_from(real_assoc.to_numpy(), k=1)
    corr_mae = float(np.abs(real_assoc.to_numpy()[upper] - synth_assoc.to_numpy()[upper]).mean())
    distinguish_auc = synthetic_real_classifier_auc(test_df, synth_df)
    fidelity_rows.append({
        "model": name,
        "mean_js_distance": float(np.mean(list(js_by_col.values()))),
        "cramers_v_matrix_mae": corr_mae,
        "synthetic_vs_real_auc": distinguish_auc,
    })
    fidelity_details[name] = {"js_by_column": js_by_col, "real_assoc": real_assoc, "synth_assoc": synth_assoc}

fidelity_df = pd.DataFrame(fidelity_rows)
display(fidelity_df.sort_values("model"))
fidelity_df.to_csv(OUTPUT_DIR / "fidelity_metrics.csv", index=False)

# ## 18. Downstream utility: TSTR (train on synthetic, test on real)
#
# For every generator, train a new Logistic Regression classifier on generated rows only and evaluate it on the same held-out real test respondents. Compare with the TRTR real-training baseline. If a synthetic sample collapses to one target class, the notebook records that run as an evaluation failure instead of inventing a score.

utility_rows = [trtr_metrics]
for name, synth_df in synthetic_datasets.items():
    try:
        synth_binary = binary_eligible(synth_df)
        if len(synth_binary) == 0 or synth_binary[TARGET_COL].nunique() < 2:
            raise ValueError(f"{name}: generated data has insufficient clear yes/no target classes.")
        metrics, _ = train_evaluate_tstr(synth_binary, binary_test_real, name)
        utility_rows.append(metrics)
    except ValueError as exc:
        print("Utility evaluation warning:", exc)
        utility_rows.append({"model": name, "accuracy": np.nan, "precision": np.nan,
                             "recall": np.nan, "f1": np.nan, "roc_auc": np.nan})

utility_df = pd.DataFrame(utility_rows)
display(utility_df)
utility_df.to_csv(OUTPUT_DIR / "tstr_utility_metrics.csv", index=False)

# ## 19. Empirical privacy audits
#
# We include three audits:
#
# 1. **DCR (distance to closest record):** measure each synthetic record's nearest Euclidean distance to a subsample of training respondents and to a subsample of held-out respondents in one-hot feature space. A much smaller distance to training data than test data can warrant investigation. DCR is not a formal DP proof.
# 2. **Exact-match rate:** percentage of synthetic rows exactly matching a complete training/test categorical row. Repetition can occur naturally in low-cardinality survey data, so this is a warning indicator, not proof of memorization.
# 3. **Loss-based membership-inference audit:** compare diffusion denoising losses on sampled training members and held-out non-members; use negative loss as the membership score and compute attack ROC-AUC. AUC near 0.5 means this particular attack is near random, not that all attacks fail.
#
# Because the 2025 BRFSS release is public, these audits are methodological demonstrations on a public benchmark. Do not interpret them as a privacy assessment of a confidential hospital deployment.

def as_code_tuple(frame: pd.DataFrame, columns=ALL_COLS):
    normalized = frame.loc[:, columns].apply(
        lambda col: col.map(lambda v: normalize_code(v, col.name))
    )
    return set(map(tuple, normalized.to_numpy(dtype=object)))


def exact_match_rates(synth_df, real_train_df, real_test_df):
    train_set = as_code_tuple(real_train_df)
    test_set = as_code_tuple(real_test_df)
    synth_rows = list(map(tuple, synth_df[ALL_COLS].to_numpy(dtype=object)))
    train_rate = float(np.mean([row in train_set for row in synth_rows]))
    test_rate = float(np.mean([row in test_set for row in synth_rows]))
    return train_rate, test_rate


def nearest_distance_summary(synth_df, train_records, test_records):
    rng = np.random.default_rng(SEED)
    n_syn = min(DCR_SYNTH_N, len(synth_df))
    n_train = min(DCR_REFERENCE_N, len(train_records))
    n_test = min(DCR_REFERENCE_N, len(test_records))
    synth_part = synth_df.sample(n=n_syn, random_state=SEED)
    train_part = train_records.sample(n=n_train, random_state=SEED)
    test_part = test_records.sample(n=n_test, random_state=SEED)

    X_syn = encode_frame(synth_part, ALL_COLS)
    X_train_ref = encode_frame(train_part, ALL_COLS)
    X_test_ref = encode_frame(test_part, ALL_COLS)

    train_nn = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute", n_jobs=-1)
    test_nn = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute", n_jobs=-1)
    train_nn.fit(X_train_ref)
    test_nn.fit(X_test_ref)
    d_train = train_nn.kneighbors(X_syn, return_distance=True)[0].ravel()
    d_test = test_nn.kneighbors(X_syn, return_distance=True)[0].ravel()
    return {
        "dcr_train_median": float(np.median(d_train)),
        "dcr_test_median": float(np.median(d_test)),
        "dcr_train_p05": float(np.quantile(d_train, 0.05)),
        "dcr_test_p05": float(np.quantile(d_test, 0.05)),
        "dcr_median_ratio_train_test": float(np.median(d_train) / (np.median(d_test) + 1e-12)),
    }


@torch.no_grad()
def denoising_loss_scores(model, encoded_array, max_records=MIA_MAX_PER_GROUP, repeats=3):
    model.eval()
    rng = np.random.default_rng(SEED)
    if len(encoded_array) > max_records:
        idx = rng.choice(len(encoded_array), size=max_records, replace=False)
        encoded_array = encoded_array[idx]
    tensor = torch.from_numpy(np.asarray(encoded_array, dtype=np.float32))
    score_parts = []
    eval_batch = 512
    for start in range(0, len(tensor), eval_batch):
        x0 = tensor[start:start+eval_batch].to(DEVICE)
        accumulated = torch.zeros(x0.shape[0], device=DEVICE)
        for _ in range(repeats):
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, noise)
            pred = model(x_t, t)
            accumulated += (pred - noise).pow(2).mean(dim=1)
        score_parts.append((accumulated / repeats).cpu().numpy())
    return np.concatenate(score_parts)


def membership_inference_auc(model, x_train, x_test):
    train_scores = denoising_loss_scores(model, x_train)
    test_scores = denoising_loss_scores(model, x_test)
    labels = np.r_[np.ones(len(train_scores), dtype=int), np.zeros(len(test_scores), dtype=int)]
    scores = np.r_[-train_scores, -test_scores]  # lower loss -> higher membership score
    return float(roc_auc_score(labels, scores)), float(np.mean(train_scores)), float(np.mean(test_scores))


privacy_audit_rows = []
for name, synth_df in synthetic_datasets.items():
    print("Privacy audit:", name)
    dcr = nearest_distance_summary(synth_df, train_df, test_df)
    exact_train, exact_test = exact_match_rates(synth_df, train_df, test_df)
    mia_auc, member_loss, nonmember_loss = membership_inference_auc(
        models[name]["model"], X_train_tab, X_test_tab
    )
    privacy_audit_rows.append({
        "model": name,
        **dcr,
        "exact_train_match_rate": exact_train,
        "exact_test_match_rate": exact_test,
        "loss_mia_auc": mia_auc,
        "mean_member_denoising_loss": member_loss,
        "mean_nonmember_denoising_loss": nonmember_loss,
    })

privacy_audit_df = pd.DataFrame(privacy_audit_rows)
display(privacy_audit_df)
privacy_audit_df.to_csv(OUTPUT_DIR / "empirical_privacy_audits.csv", index=False)

# ## 20. Combine all measured results
#
# This table joins the privacy accountant with utility and fidelity metrics. The non-private baseline has no finite DP epsilon and must be shown as `∞` / not protected by DP. Do not replace missing experiment outputs with illustrative numbers.

results_df = (
    fidelity_df
    .merge(utility_df, on="model", how="left")
    .merge(privacy_audit_df, on="model", how="left")
)
if not privacy_df.empty:
    results_df = results_df.merge(
        privacy_df[["model", "target_epsilon", "achieved_epsilon", "delta", "noise_multiplier"]],
        on="model", how="left"
    )
else:
    results_df["target_epsilon"] = np.nan
    results_df["achieved_epsilon"] = np.nan

results_df.loc[results_df["model"] == "non_private", "target_epsilon"] = np.inf
results_df.loc[results_df["model"] == "non_private", "achieved_epsilon"] = np.inf
results_df.to_csv(OUTPUT_DIR / "combined_experiment_results.csv", index=False)
display(results_df)

# ## 21. Visualizations for the research report
#
# The following plots are generated from actual notebook results: training loss, per-feature distribution distance, utility versus privacy budget, synthetic-vs-real distinguishability, membership-inference AUC, and a real-versus-synthetic association heatmap. A lower JS / association error is better; TSTR F1 and ROC-AUC should be compared against the TRTR baseline; a membership-attack AUC near 0.5 is preferable for the tested attack.

# 21.1 Training curves
plt.figure(figsize=(9, 5))
for name, history in training_histories.items():
    plt.plot(range(1, len(history)+1), history, marker="o", label=name)
plt.xlabel("Epoch")
plt.ylabel("Mean denoising MSE")
plt.title("Diffusion training loss")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "training_loss.png", dpi=180)
plt.show()

# 21.2 Utility / fidelity / privacy plots
plot_df = results_df.copy()
plot_df["epsilon_plot"] = plot_df["achieved_epsilon"]
plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
private_plot = plot_df[plot_df["model"].str.startswith("dp_")].sort_values("achieved_epsilon")
axes[0].plot(private_plot["achieved_epsilon"], private_plot["f1"], marker="o")
axes[0].set_xlabel("Achieved epsilon (lower = stronger DP)")
axes[0].set_ylabel("TSTR F1")
axes[0].set_title("Privacy budget vs downstream utility")
axes[0].grid(alpha=0.25)

axes[1].plot(private_plot["achieved_epsilon"], private_plot["mean_js_distance"], marker="o")
axes[1].set_xlabel("Achieved epsilon")
axes[1].set_ylabel("Mean JS distance (lower is better)")
axes[1].set_title("Privacy budget vs distribution fidelity")
axes[1].grid(alpha=0.25)

axes[2].plot(private_plot["achieved_epsilon"], private_plot["loss_mia_auc"], marker="o")
axes[2].axhline(0.5, linestyle="--", linewidth=1, label="Random-guess reference")
axes[2].set_xlabel("Achieved epsilon")
axes[2].set_ylabel("Membership inference AUC")
axes[2].set_title("Privacy budget vs tested attack")
axes[2].legend()
axes[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "privacy_utility_tradeoff.png", dpi=180)
plt.show()

# 21.3 Per-feature JS distances for all models
js_frame = pd.DataFrame({name: detail["js_by_column"] for name, detail in fidelity_details.items()})
plt.figure(figsize=(12, 7))
sns.heatmap(js_frame, annot=False, cmap="mako", cbar_kws={"label": "Jensen-Shannon distance"})
plt.title("Per-feature categorical distribution distance")
plt.xlabel("Generator")
plt.ylabel("BRFSS variable")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plots" / "per_feature_js_heatmap.png", dpi=180)
plt.show()

# 21.4 Cramer's V heatmaps for one private model (epsilon closest to 3)
dp_names = [name for name in synthetic_datasets if name.startswith("dp_eps_")]
if dp_names:
    selected_model = min(dp_names, key=lambda n: abs(models[n]["achieved_epsilon"] - 3.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(real_assoc, vmin=0, vmax=1, cmap="viridis", ax=axes[0])
    axes[0].set_title("Real training data – Cramér's V")
    sns.heatmap(fidelity_details[selected_model]["synth_assoc"], vmin=0, vmax=1, cmap="viridis", ax=axes[1])
    axes[1].set_title(f"Synthetic ({selected_model}) – Cramér's V")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "association_heatmaps.png", dpi=180)
    plt.show()

print("Plots saved in:", OUTPUT_DIR / "plots")

# ## 22. Code-domain validity and generated-data diagnostics
#
# Block-wise argmax guarantees that every decoded value belongs to the predeclared category domain or to the fixed `__MISSING__` / `__OTHER__` sentinel. It does **not** guarantee realistic cross-feature constraints, causal consistency, or clinical plausibility. We report target balance and the frequency of sentinel categories to help interpret generated data.

diagnostic_rows = []
for name, synth_df in synthetic_datasets.items():
    sentinel_rates = {}
    for col in FEATURE_COLS:
        sentinel_rates[col] = synth_df[col].isin([MISSING_TOKEN, OTHER_TOKEN]).mean()
    diagnostic_rows.append({
        "model": name,
        "n_synthetic": len(synth_df),
        "diabetes_yes_rate": (synth_df[TARGET_COL] == "1").mean(),
        "diabetes_no_rate": (synth_df[TARGET_COL] == "3").mean(),
        "mean_feature_sentinel_rate": float(np.mean(list(sentinel_rates.values()))),
        "max_feature_sentinel_rate": float(np.max(list(sentinel_rates.values()))),
    })

diagnostics_df = pd.DataFrame(diagnostic_rows)
display(diagnostics_df)
diagnostics_df.to_csv(OUTPUT_DIR / "synthetic_data_diagnostics.csv", index=False)

# ## 23. Report-ready interpretation helper
#
# This cell creates a Markdown summary populated from the **measured** notebook output. Read and edit the generated interpretation before including it in the final academic report. Do not call the highest-utility model “best” without considering the privacy budget, fidelity, attack audit, and limitations.

def fmt_metric(value, digits=4):
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"

summary_lines = [
    "# Experimental summary – 2025 CDC BRFSS DP diffusion project",
    "",
    f"- CDC source: {CDC_PAGE}",
    f"- Rows in downloaded source file / sampled rows: {source_rows:,} / {len(data):,}",
    f"- Training respondents: {len(train_df):,}; held-out respondents: {len(test_df):,}",
    f"- Generator target: all DIABETE4 response codes; downstream binary task uses code 1 (yes) vs code 3 (no)",
    f"- Diffusion steps: {TIMESTEPS}; epochs: {EPOCHS}; batch size: {BATCH_SIZE}",
    f"- Delta: {DELTA:.3e}",
    "",
    "## Measured results",
    "",
    "| Model | Achieved ε | TSTR F1 | TSTR ROC-AUC | Mean JS distance | Synthetic-vs-real AUC | Loss-MIA AUC |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for _, row in results_df.iterrows():
    eps = "∞ (non-private)" if row["model"] == "non_private" else fmt_metric(row.get("achieved_epsilon", np.nan), 3)
    summary_lines.append(
        f"| {row['model']} | {eps} | {fmt_metric(row.get('f1', np.nan))} | "
        f"{fmt_metric(row.get('roc_auc', np.nan))} | {fmt_metric(row.get('mean_js_distance', np.nan))} | "
        f"{fmt_metric(row.get('synthetic_vs_real_auc', np.nan))} | {fmt_metric(row.get('loss_mia_auc', np.nan))} |"
    )
summary_lines += [
    "",
    "## Interpretation prompts (complete after reviewing measured output)",
    "",
    "1. Which DP budget produced the most useful TSTR model, and how close was it to the TRTR real-data baseline?",
    "2. Did the JS and Cramér's V errors increase as the privacy budget became stricter? Were there exceptions?",
    "3. Was the membership-inference AUC near 0.5 for any model? What does this tested attack measure, and what does it not prove?",
    "4. Did DCR training and test distances differ materially? Investigate very small nearest-neighbour distances and exact matches.",
    "5. What are the limitations of public, self-reported BRFSS data, unweighted training, one-hot Gaussian diffusion, and a single attack implementation?",
    "",
    "## Required caution",
    "",
    "Report the achieved epsilon from Opacus, not merely the requested target epsilon. Empirical attack metrics do not replace the formal DP accountant. BRFSS is public survey data and this experiment does not establish privacy for a real private hospital database."
]
summary_text = "\n".join(summary_lines)
summary_path = OUTPUT_DIR / "report_ready_summary.md"
summary_path.write_text(summary_text, encoding="utf-8")
print(summary_text)

# ## 24. Export all outputs
#
# The notebook saves individual CSVs, checkpoints, plots, and a report-ready Markdown summary. The final ZIP can be downloaded from Colab's file browser or via the cell below. For persistent storage, mount Google Drive before the run and set `OUTPUT_DIR` to a Drive folder in the configuration cell.

# Save configuration and run metadata.
run_metadata = {
    "dataset": "CDC 2025 BRFSS annual combined landline + cellular SAS transport file",
    "dataset_page": CDC_PAGE,
    "dataset_file_url": CDC_XPT_ZIP_URL,
    "dataset_year": 2025,
    "dataset_release_month": "August 2026",
    "source_rows_in_file": int(source_rows),
    "rows_used": int(len(data)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "privacy_unit": "one respondent record in the single-year BRFSS file",
    "generator_target": "DIABETE4 all observed response codes plus fixed missing/other sentinels",
    "downstream_binary_target": "DIABETE4 code 1 yes vs code 3 no; other responses excluded only from binary evaluation",
    "seed": SEED,
    "device": str(DEVICE),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "timesteps": TIMESTEPS,
    "target_epsilons": TARGET_EPSILONS,
    "delta": DELTA,
    "max_grad_norm": MAX_GRAD_NORM,
    "secure_mode": SECURE_MODE,
    "feature_columns": FEATURE_COLS,
    "target_column": TARGET_COL,
    "note": "Public self-reported survey benchmark; not hospital EHR data; unweighted generative training."
}
(OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

# Bundle the experiment artifacts, excluding the large raw CDC ZIP by default.
import shutil
bundle_path = OUTPUT_DIR / "DP_Medical_Synthetic_2025_BRFSS_Results.zip"
with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file() or path == bundle_path:
            continue
        if path.name == ZIP_PATH.name or path.name.endswith(".part"):
            continue
        zf.write(path, arcname=path.relative_to(OUTPUT_DIR))

print("Artifacts saved to:", OUTPUT_DIR.resolve())
print("Results bundle:", bundle_path)
print("Files:")
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        print(f" - {path.relative_to(OUTPUT_DIR)} ({path.stat().st_size/1024:.1f} KB)")

# ## 25. Limitations, responsible use, and conclusion
#
# ### Limitations to include in the report
#
# 1. **Dataset type:** BRFSS is self-reported population health survey data, not a clinical electronic health record. It contains one annual record per respondent and does not represent longitudinal patient journeys.
# 2. **Public benchmark:** the source dataset is publicly released. The DP result is a guarantee for the model-training mechanism relative to participation in this benchmark, not a guarantee that the original public survey responses are confidential.
# 3. **Survey design:** BRFSS uses a complex sample design and survey weights. This notebook does not use weights in generative training; the synthetic data should not be used to estimate population prevalence.
# 4. **Simplified diffusion:** Gaussian DDPM over one-hot encoded categories is a practical baseline. It is not a native mixed-type diffusion model such as the complete TabDDPM/TabDiff approaches.
# 5. **Clinical validity:** valid individual codes do not guarantee realistic cross-variable or clinically meaningful combinations. Synthetic records are not for diagnosis, treatment, or clinical decision-making.
# 6. **Privacy audit scope:** loss-based membership inference, DCR, and duplicate rates are empirical tests. Failure to find leakage in these tests does not prove privacy.
# 7. **Preprocessing and postprocessing:** the schema is fixed from public documentation. Any additional private-data-derived feature selection, hyperparameter tuning, filtering, or published summary may need its own privacy analysis.
# 8. **Randomness:** this research notebook sets Opacus `secure_mode=False` for compatibility; Opacus documents that secure cryptographic randomness requires `secure_mode=True` and `torchcsprng`. The reported accountant epsilon is a theoretical mechanism-accounting result for the research run, not production deployment certification.
#
# ### Conclusion template
#
# After running the experiments, summarize whether the measured results support the hypotheses. Report the achieved epsilon/delta for every DP model, compare TSTR metrics with the TRTR baseline, discuss distribution and association errors, and interpret empirical privacy audits cautiously. State explicitly whether the observed utility gains at larger epsilon justify the weaker privacy bound for the intended research use case.
#
# ### Next research extensions
#
# - Implement or adapt a native mixed-type categorical/continuous diffusion architecture and account for its complete training pipeline with DP-SGD.
# - Compare with non-diffusion synthetic tabular baselines such as CTGAN/TVAE, including private variants where available.
# - Add conditional generation for target-controlled diabetes cohorts and evaluate whether conditioning changes privacy accounting.
# - Use secure randomness, patient/household-level adjacency when relevant, and a formally reviewed pipeline on data that is genuinely confidential and authorized for research.

# ## 26. References
#
# 1. Centers for Disease Control and Prevention (CDC). **2025 BRFSS Survey Data and Documentation.** https://www.cdc.gov/brfss/annual_data/annual_2025.html (2025 annual file; released August 2026; page last reviewed September 9, 2026).
# 2. CDC. **2025 Combined Landline and Telephone Multiple Data Variable Layout.** https://www.cdc.gov/brfss/annual_data/2025/llcp_varlayout_25_onecolumn.html
# 3. Kotelnikov, A. et al. **TabDDPM: Modelling Tabular Data with Diffusion Models.** ICML 2023. https://arxiv.org/abs/2209.15421
# 4. Shi, J. et al. **TabDiff: A Mixed-type Diffusion Model for Tabular Data.** ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html
# 5. Opacus. **PrivacyEngine API documentation.** https://opacus.ai/api/privacy_engine.html
# 6. Abadi, M. et al. **Deep Learning with Differential Privacy.** CCS 2016. https://arxiv.org/abs/1607.00133
#
# ---
#
# **End of notebook.** All reported numbers must be generated by executing the experiment cells in this notebook.
