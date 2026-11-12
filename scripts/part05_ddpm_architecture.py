#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part5_DP_Medical_DDPM_Architecture.ipynb.

Run from the project root with: python scripts/part05_ddpm_architecture.py
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
print("Part 05 – Part5_DP_Medical_DDPM_Architecture")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 5 – Tabular DDPM Architecture and Fixed-Feature Encoding
#
# **Project:** Privacy-Preserving Synthetic Health Data Generation with Differentially Private Diffusion Models  
# **Dataset:** CDC 2025 Behavioral Risk Factor Surveillance System (BRFSS)  
# **Environment:** Google Colab / Python 3  
# **Input:** `data/train.csv` and `data/test.csv` created by Part 2  
# **Output:** fixed one-hot arrays, schema manifest, model configuration and architecture diagnostics.
#
# ## What this part does
#
# 1. Reopens the prepared train/test respondent split from Google Drive.
# 2. Applies a fixed, predeclared category schema to 23 predictors and `DIABETE4`.
# 3. Encodes each categorical field as a separate one-hot block and validates every row.
# 4. Saves the arrays and schema so later notebooks do not refit encoders or depend on runtime variables.
# 5. Defines the time-conditioned denoising MLP, cosine schedule and forward noising process.
#
# **Run order:** Complete Parts 1–4 first. This notebook does not download or modify the raw CDC file.
#
# > Method note: the model is a compact Gaussian DDPM over a fixed one-hot representation of tabular variables. It is DDPM-inspired and should not be described as an exact reproduction of a specialised method such as TabDDPM.

# ## 1. Install / import dependencies
#
# Colab normally includes PyTorch. This cell installs only the tabular and plotting dependencies if needed.

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

# ## 2. Load the prepared respondent split
#
# The train/test split was created in Part 2 before model fitting. The same split is reused by all later experiments to keep comparisons consistent.

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Missing prepared data. Run Part 2 and confirm train.csv and test.csv exist in {DATA_DIR}"
    )
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df.columns = [str(c).upper() for c in train_df.columns]
test_df.columns = [str(c).upper() for c in test_df.columns]
print('Train shape:', train_df.shape, '| Test shape:', test_df.shape)
display(train_df.head())

# ## 3. Declare the fixed public category schema
#
# The allowed code lists are fixed from the BRFSS variable layout rather than learned from the training sample. Missing and out-of-domain predictor values are mapped to explicit sentinel categories. For the diabetes response, unexpected non-missing codes trigger an error so they are reviewed rather than silently absorbed.

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
TARGET_DOMAIN = ["1", "2", "3", "4", "7", "9"]
FEATURE_COLS = list(FEATURE_DOMAINS.keys())
ALL_COLS = FEATURE_COLS + [TARGET_COL]
MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"
DOMAINS = {
    col: [str(v) for v in FEATURE_DOMAINS[col]] + [MISSING_TOKEN, OTHER_TOKEN]
    for col in FEATURE_COLS
}
DOMAINS[TARGET_COL] = TARGET_DOMAIN + [MISSING_TOKEN, OTHER_TOKEN]
CATEGORY_TO_INDEX = {col: {value: idx for idx, value in enumerate(DOMAINS[col])} for col in ALL_COLS}


def normalize_code(value, column):
    """Normalize an input code to the public, fixed category schema."""
    if isinstance(value, str) and value in CATEGORY_TO_INDEX[column]:
        return value
    if pd.isna(value):
        return MISSING_TOKEN
    try:
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            return OTHER_TOKEN if column != TARGET_COL else None
        code = str(int(numeric))
    except (TypeError, ValueError):
        return OTHER_TOKEN if column != TARGET_COL else None
    if code in CATEGORY_TO_INDEX[column]:
        return code
    return OTHER_TOKEN if column != TARGET_COL else None


def encode_frame(frame, columns=ALL_COLS):
    blocks = []
    for col in columns:
        vals = frame[col].map(lambda v: normalize_code(v, col)).tolist()
        if any(v is None for v in vals):
            raise ValueError(f"Unexpected DIABETE4 category found in {col}; inspect source codes before encoding.")
        idx = np.fromiter((CATEGORY_TO_INDEX[col][v] for v in vals), dtype=np.int64, count=len(vals))
        blocks.append(np.eye(len(DOMAINS[col]), dtype=np.float32)[idx])
    return np.concatenate(blocks, axis=1).astype(np.float32)

missing_columns = [c for c in ALL_COLS if c not in train_df.columns or c not in test_df.columns]
if missing_columns:
    raise KeyError(f"Required columns are missing from the Part 2 files: {missing_columns}")
print('Predictor count:', len(FEATURE_COLS))
print('Total categorical fields including target:', len(ALL_COLS))
print('DIABETE4 values in train:')
display(train_df[TARGET_COL].value_counts(dropna=False).sort_index().to_frame('count'))

# ## 4. Encode all fields and validate one-hot blocks
#
# Each respondent becomes a concatenation of one-hot category blocks. Exactly one position must be active in each field block, so the sum across a full row must equal the number of fields (24).

X_train_tab = encode_frame(train_df, ALL_COLS)
X_test_tab = encode_frame(test_df, ALL_COLS)
INPUT_DIM = int(X_train_tab.shape[1])

feature_slices = {}
cursor = 0
for col in ALL_COLS:
    width = len(DOMAINS[col])
    feature_slices[col] = [cursor, cursor + width]
    cursor += width

assert X_train_tab.shape[0] == len(train_df)
assert X_test_tab.shape[0] == len(test_df)
assert np.allclose(X_train_tab.sum(axis=1), len(ALL_COLS))
assert np.allclose(X_test_tab.sum(axis=1), len(ALL_COLS))
assert np.isfinite(X_train_tab).all() and np.isfinite(X_test_tab).all()
print('Encoded train:', X_train_tab.shape)
print('Encoded test :', X_test_tab.shape)
print('Input dimension:', INPUT_DIM)
print('Row one-hot sum check:', np.unique(X_train_tab.sum(axis=1)))

# ## 5. Persist encoded data and schema manifest
#
# The `.npy` arrays avoid repeating category conversion in later sessions. The JSON manifest documents the exact category order and feature-block boundaries used by the saved arrays.

np.save(DATA_DIR / 'encoded_train.npy', X_train_tab)
np.save(DATA_DIR / 'encoded_test.npy', X_test_tab)

schema_manifest = {
    'dataset': 'CDC 2025 BRFSS',
    'privacy_unit': 'one respondent row',
    'feature_columns': FEATURE_COLS,
    'target_column': TARGET_COL,
    'all_columns': ALL_COLS,
    'domains': DOMAINS,
    'feature_slices': feature_slices,
    'input_dim': INPUT_DIM,
    'train_rows': int(len(train_df)),
    'test_rows': int(len(test_df)),
    'encoding': 'fixed one-hot blocks in ALL_COLS order',
    'missing_token': MISSING_TOKEN,
    'other_token': OTHER_TOKEN,
    'seed': SEED,
}
(DATA_DIR / 'schema_manifest.json').write_text(json.dumps(schema_manifest, indent=2), encoding='utf-8')

data_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(train_df[ALL_COLS], index=False).values.tobytes()).hexdigest()
config = {
    'model_name': 'TabularDenoiser_Gaussian_DDPM',
    'timesteps': 100,
    'hidden_dim': 128,
    'time_embed_dim': 32,
    'batch_size': 256,
    'epochs': 5,
    'learning_rate': 0.0002,
    'max_grad_norm': 1.0,
    'target_epsilons': [1.0, 3.0, 8.0],
    'accountant': 'prv',
    'secure_mode': False,
    'seed': SEED,
    'input_dim': INPUT_DIM,
    'train_rows': int(len(train_df)),
    'test_rows': int(len(test_df)),
    'data_fingerprint': data_fingerprint,
    'note': 'Defaults are sized for a modest Colab run; secure_mode=False is not cryptographic RNG hardening.'
}
(DATA_DIR / 'ddpm_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
print('Saved encoded arrays, schema_manifest.json and ddpm_config.json')
print('Training-data fingerprint:', data_fingerprint)

# ## 6. Define the time-conditioned denoiser and forward diffusion process
#
# For a clean encoded vector $x_0$, the forward process adds Gaussian noise according to a time-dependent schedule. The model receives the noisy vector and timestep and predicts the added noise. The loss is mean-squared noise prediction error, computed per respondent.

with open(DATA_DIR / 'ddpm_config.json', encoding='utf-8') as f:
    config = json.load(f)
TIMESTEPS = int(config['timesteps'])
HIDDEN_DIM = int(config['hidden_dim'])
TIME_EMBED_DIM = int(config['time_embed_dim'])
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class TabularDenoiser(nn.Module):
    """Compact timestep-conditioned MLP noise predictor for one-hot tabular rows."""
    def __init__(self, input_dim, time_dim=TIME_EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.time_dim = int(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(int(input_dim) + self.time_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(input_dim)),
        )

    def forward(self, x_t, t):
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        return self.network(torch.cat([x_t, t_emb], dim=1))


def cosine_noise_schedule(timesteps, device):
    steps = int(timesteps) + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alpha_bar = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = torch.clamp(1.0 - (alpha_bar[1:] / alpha_bar[:-1]), min=1e-5, max=0.999)
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


def q_sample(x0, t, noise, schedule):
    alpha_bar_t = schedule["alpha_bars"][t].unsqueeze(1)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def diffusion_per_example_loss(model, x0, schedule, timesteps):
    """Returns one scalar loss per respondent; required by Opacus DP-SGD."""
    batch_size = x0.shape[0]
    t = torch.randint(0, timesteps, (batch_size,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    predicted_noise = model(x_t, t)
    return (predicted_noise - noise).pow(2).mean(dim=1)

SCHEDULE = cosine_noise_schedule(TIMESTEPS, DEVICE)
print('Parameters:', sum(p.numel() for p in TabularDenoiser(INPUT_DIM).parameters()))
print('Final cumulative alpha:', float(SCHEDULE['alpha_bars'][-1].detach().cpu()))

# ## 7. Architecture and noise-schedule diagnostics
#
# These figures are generated from the configured architecture and mathematical schedule, not measured experiment outcomes.

# Verify forward noising and model output shapes.
model_check = TabularDenoiser(INPUT_DIM).to(DEVICE)
x0_check = torch.from_numpy(X_train_tab[:8]).to(DEVICE)
t_check = torch.randint(0, TIMESTEPS, (len(x0_check),), device=DEVICE)
noise_check = torch.randn_like(x0_check)
xt_check = q_sample(x0_check, t_check, noise_check, SCHEDULE)
pred_check = model_check(xt_check, t_check)
assert pred_check.shape == x0_check.shape
assert torch.isfinite(pred_check).all()
print('Forward process shape:', tuple(xt_check.shape))
print('Denoiser output shape:', tuple(pred_check.shape))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(SCHEDULE['alpha_bars'].detach().cpu().numpy())
axes[0].set_title('Cosine cumulative signal level')
axes[0].set_xlabel('Diffusion timestep')
axes[0].set_ylabel('Cumulative alpha')
axes[1].plot(SCHEDULE['betas'].detach().cpu().numpy())
axes[1].set_title('Cosine noise variance schedule')
axes[1].set_xlabel('Diffusion timestep')
axes[1].set_ylabel('Beta')
for ax in axes: ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(PLOTS_DIR / 'part5_ddpm_noise_schedule.png', dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis('off')
items = [('Noisy row $x_t$', 0.08), ('Time embedding $t$', 0.32), ('Concatenate', 0.56), ('MLP + SiLU', 0.76), ('Predicted noise $\\epsilon_\\theta$', 0.94)]
for label, x in items:
    ax.text(x, 0.56, label, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#eef3f8', edgecolor='#42617a'))
for x1, x2 in [(0.16,0.25),(0.40,0.48),(0.64,0.69),(0.84,0.87)]:
    ax.annotate('', xy=(x2,0.56), xytext=(x1,0.56), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.set_title('Time-conditioned tabular denoiser (conceptual architecture)')
fig.savefig(PLOTS_DIR / 'part5_denoiser_architecture.png', dpi=180, bbox_inches='tight')
plt.show()

# ## 8. Part 5 completion checklist
#
# Expected persistent files:
#
# - `data/encoded_train.npy`
# - `data/encoded_test.npy`
# - `data/schema_manifest.json`
# - `data/ddpm_config.json`
# - `plots/part5_ddpm_noise_schedule.png`
# - `plots/part5_denoiser_architecture.png`
#
# Do not edit the category order manually after these files are created. If the feature list changes, rerun Part 5 and then retrain Parts 6–8.
