#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part9_DP_Medical_Synthetic_Data_Generation.ipynb.

Run from the project root with: python scripts/part09_synthetic_generation.py
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
print("Part 09 – Part9_DP_Medical_Synthetic_Data_Generation")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 9 – Synthetic Respondent Generation with the Trained DDPM
#
# **Project:** Privacy-Preserving Synthetic Health Data Generation with Differentially Private Diffusion Models  
# **Dataset:** CDC 2025 BRFSS respondent-level survey data  
# **Input:** Part 5 schema/configuration and a completed model checkpoint from Part 6 or Part 7  
# **Output:** decoded synthetic CSV, generation metadata and report-ready plots saved to Google Drive.
#
# ## Purpose and run order
#
# This notebook samples from the learned reverse diffusion process, then converts each continuous generated row back into valid categorical fields by taking the `argmax` within each fixed one-hot block. It does not train a model. Run Parts 5 and 6, or Parts 5 and 7, before this part. Part 8 is recommended before describing a DP model's achieved privacy budget.
#
# **Important interpretation:** the model is a Gaussian DDPM over one-hot encoded tabular rows. The categorical argmax is a decoding step; it does not make the underlying diffusion process a specialised categorical diffusion model. Synthetic records are research artifacts, not real patient records or clinical advice.

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

import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print('Compute device:', DEVICE)

# ## 2. Validate required encoded-data and model artifacts

ENC_TRAIN_PATH = DATA_DIR / 'encoded_train.npy'
ENC_TEST_PATH = DATA_DIR / 'encoded_test.npy'
if not ENC_TRAIN_PATH.exists() or not ENC_TEST_PATH.exists():
    raise FileNotFoundError('Run Part 5 to create encoded_train.npy and encoded_test.npy.')
X_train = np.load(ENC_TRAIN_PATH, mmap_mode='r')
X_test = np.load(ENC_TEST_PATH, mmap_mode='r')
if X_train.ndim != 2 or X_train.shape[1] != INPUT_DIM:
    raise ValueError(f'Encoded training shape {X_train.shape} does not match schema input_dim={INPUT_DIM}.')
if int(config.get('train_rows', -1)) != X_train.shape[0]:
    raise ValueError('The encoded training data size differs from the saved DDPM configuration.')
print('Encoded train:', X_train.shape, '| encoded test:', X_test.shape)
print('Training-data fingerprint:', config.get('data_fingerprint'))

# ## 3. Recreate the exact trained denoiser and cosine schedule
#
# The class definition and schedule below intentionally match Part 5/6/7. Changing layer sizes, timestep count or embedding dimensions would make the saved state dictionary incompatible.

import math

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding

class TabularDenoiser(nn.Module):
    def __init__(self, input_dim, time_dim=None, hidden_dim=None):
        super().__init__()
        self.time_dim = int(time_dim or config['time_embed_dim'])
        hidden_dim = int(hidden_dim or config['hidden_dim'])
        self.time_mlp = nn.Sequential(nn.Linear(self.time_dim, self.time_dim), nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))
        self.network = nn.Sequential(
            nn.Linear(int(input_dim) + self.time_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, int(input_dim)))
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
    return {'betas': betas, 'alphas': alphas, 'alpha_bars': alpha_bars,
            'alpha_bars_prev': alpha_bars_prev, 'posterior_variance': posterior_variance}

TIMESTEPS = int(config['timesteps'])
SCHEDULE = cosine_noise_schedule(TIMESTEPS, DEVICE)
print('Timesteps:', TIMESTEPS, '| input dimension:', INPUT_DIM)

# ## 4. Select a checkpoint to generate from
#
# Set `MODEL_KIND` to `"dp"` or `"non_private"`. For a DP model, set `SELECTED_EPSILON` to one of the completed Part 7 values. The default is the ε=3 run. This choice controls which model is sampled; it does not retrain it.

MODEL_KIND = 'dp'                 # 'dp' or 'non_private'
SELECTED_EPSILON = 3.0            # used only when MODEL_KIND == 'dp'
FORCE_REGENERATE = False          # True overwrites the selected synthetic CSV

if MODEL_KIND == 'non_private':
    MODEL_PATH = CHECKPOINT_DIR / 'ddpm_non_private.pt'
    MODEL_LABEL = 'non_private'
    privacy_meta = None
elif MODEL_KIND == 'dp':
    eps_tag = str(float(SELECTED_EPSILON)).replace('.', 'p')
    MODEL_PATH = CHECKPOINT_DIR / f'ddpm_dp_eps_{eps_tag}.pt'
    MODEL_LABEL = f'dp_eps_{eps_tag}'
    META_PATH = METRICS_DIR / f'dp_run_eps_{eps_tag}.json'
    if not META_PATH.exists():
        raise FileNotFoundError(f'Missing {META_PATH.name}. Complete Part 7 for epsilon={SELECTED_EPSILON:g} first.')
    privacy_meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    if privacy_meta.get('status') != 'completed':
        raise ValueError('Part 7 metadata does not mark this model as completed.')
    if privacy_meta.get('data_fingerprint') != config.get('data_fingerprint'):
        raise ValueError('The DP model was not trained using the current Part 5 dataset fingerprint.')
else:
    raise ValueError("MODEL_KIND must be either 'dp' or 'non_private'.")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Model checkpoint not found: {MODEL_PATH}. Complete Part 6 or Part 7 first.')
if privacy_meta is not None:
    print('Selected model:', MODEL_LABEL, '| target epsilon:', privacy_meta['target_epsilon'],
          '| achieved epsilon:', privacy_meta['achieved_epsilon'], '| delta:', privacy_meta['delta'])
else:
    print('Selected model:', MODEL_LABEL, '| formal DP guarantee: NONE (non-private baseline)')

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if checkpoint.get('data_fingerprint') != config.get('data_fingerprint'):
    raise ValueError('Model checkpoint fingerprint does not match the current encoded training set.')
if int(checkpoint.get('input_dim', -1)) != INPUT_DIM:
    raise ValueError('Model checkpoint input dimension does not match schema.')
model = TabularDenoiser(INPUT_DIM).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
print('Loaded checkpoint:', MODEL_PATH.name)
print('Model parameters:', sum(p.numel() for p in model.parameters()))

# ## 5. Implement reverse diffusion sampling
#
# The sampler uses the DDPM posterior mean derived from the ε-prediction parameterisation. Fresh Gaussian noise is added for steps `t > 0`; no extra noise is added at the final step. Sampling is chunked to reduce GPU memory requirements.

@torch.no_grad()
def reverse_sample(model, n_samples, batch_size=256, seed=1234):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    generated = []
    for start in range(0, int(n_samples), int(batch_size)):
        current_batch = min(int(batch_size), int(n_samples) - start)
        x = torch.randn((current_batch, INPUT_DIM), device=DEVICE, generator=generator)
        for t_value in reversed(range(TIMESTEPS)):
            t = torch.full((current_batch,), t_value, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t)
            beta_t = SCHEDULE['betas'][t_value]
            alpha_t = SCHEDULE['alphas'][t_value]
            alpha_bar_t = SCHEDULE['alpha_bars'][t_value]
            mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise) / torch.sqrt(alpha_t)
            if t_value > 0:
                variance = SCHEDULE['posterior_variance'][t_value].clamp(min=1e-20)
                noise = torch.randn(x.shape, device=DEVICE, generator=generator)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        generated.append(x.detach().cpu().numpy().astype(np.float32))
        print(f'Generated {min(start + current_batch, n_samples):,}/{n_samples:,} rows', end='\r')
    print()
    return np.concatenate(generated, axis=0)


def decode_onehot_matrix(matrix):
    """Decode by argmax within each manifest-defined categorical field block."""
    decoded = {}
    for col in ALL_COLS:
        start, end = FEATURE_SLICES[col]
        block = matrix[:, start:end]
        indices = np.argmax(block, axis=1)
        domain = DOMAINS[col]
        decoded[col] = [domain[int(i)] for i in indices]
    return pd.DataFrame(decoded, columns=ALL_COLS)

# ## 6. Generate, decode and save synthetic respondents

# Default sample size is capped for a practical Colab run. Increase this after a successful pilot if needed.
SYNTHETIC_COUNT = min(int(X_train.shape[0]), 10_000)
GENERATION_BATCH_SIZE = 256
GENERATION_SEED = 2025

OUTPUT_PATH = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}.csv'
META_OUTPUT_PATH = SYNTHETIC_DIR / f'synthetic_{MODEL_LABEL}_metadata.json'
if OUTPUT_PATH.exists() and not FORCE_REGENERATE:
    if not META_OUTPUT_PATH.exists():
        raise FileExistsError('Synthetic CSV exists without matching metadata. Set FORCE_REGENERATE=True to rebuild it from the selected checkpoint.')
    existing_meta = json.loads(META_OUTPUT_PATH.read_text(encoding='utf-8'))
    if (existing_meta.get('data_fingerprint') != config.get('data_fingerprint')
            or existing_meta.get('model_checkpoint') != MODEL_PATH.name
            or existing_meta.get('model_label') != MODEL_LABEL):
        raise ValueError('Existing synthetic output belongs to a different data/model configuration. Set FORCE_REGENERATE=True to replace it.')
    synthetic_df = pd.read_csv(OUTPUT_PATH, dtype=str, keep_default_na=False)
    print('Reusing existing synthetic file:', OUTPUT_PATH.name, '| rows:', len(synthetic_df))
else:
    continuous_samples = reverse_sample(model, SYNTHETIC_COUNT, batch_size=GENERATION_BATCH_SIZE, seed=GENERATION_SEED)
    synthetic_df = decode_onehot_matrix(continuous_samples)
    synthetic_df.to_csv(OUTPUT_PATH, index=False)
    del continuous_samples
    print('Saved:', OUTPUT_PATH)

# Validate field presence and allowed output categories.
if list(synthetic_df.columns) != ALL_COLS:
    synthetic_df = synthetic_df[ALL_COLS]
for col in ALL_COLS:
    observed = set(synthetic_df[col].astype(str).unique())
    illegal = observed.difference(set(DOMAINS[col]))
    if illegal:
        raise ValueError(f'Illegal decoded categories in {col}: {sorted(illegal)}')

privacy_verification = None
privacy_verified = False
if privacy_meta is not None:
    verification_path = METRICS_DIR / 'privacy_accounting_verification.csv'
    if verification_path.exists():
        verification_df = pd.read_csv(verification_path)
        matched = verification_df[np.isclose(verification_df['target_epsilon'].astype(float), float(privacy_meta['target_epsilon']))]
        if len(matched):
            verification_row = matched.iloc[0]
            privacy_verified = bool(
                verification_row['verification'] == 'PASS'
                and verification_row['target_budget_status'] == 'WITHIN_TARGET'
                and abs(float(verification_row['recomputed_epsilon']) - float(privacy_meta['achieved_epsilon'])) <= max(1e-6, 1e-4 * max(1.0, float(privacy_meta['achieved_epsilon'])))
                and abs(float(verification_row['delta']) - float(privacy_meta['delta'])) <= 1e-12
                and abs(float(verification_row['noise_multiplier']) - float(privacy_meta['noise_multiplier'])) <= 1e-8
            )
            privacy_verification = {k: (v.item() if hasattr(v, 'item') else v) for k, v in verification_row.to_dict().items()}

run_metadata = {
    'dataset': config.get('model_name', 'TabularDenoiser_Gaussian_DDPM'),
    'source_dataset': 'CDC 2025 BRFSS',
    'model_label': MODEL_LABEL,
    'model_checkpoint': MODEL_PATH.name,
    'rows_generated': int(len(synthetic_df)),
    'columns': ALL_COLS,
    'generation_seed': GENERATION_SEED,
    'generation_method': 'Gaussian DDPM reverse process with argmax decoding of fixed one-hot blocks',
    'data_fingerprint': config.get('data_fingerprint'),
    'privacy_metadata': privacy_meta,
    'formal_dp_claim': bool(privacy_meta is not None and privacy_verified),
    'privacy_accounting_verification': privacy_verification,
    'note': 'The non-private checkpoint has no DP guarantee.' if MODEL_KIND == 'non_private' else ('Part 8 accountant verification passed for this selected target.' if privacy_verified else 'Part 7 metadata is present, but Part 8 verification has not passed or is not available; do not make a verified epsilon claim yet.')
}
META_OUTPUT_PATH.write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')
print('Synthetic shape:', synthetic_df.shape)
print('Saved metadata:', META_OUTPUT_PATH)
display(synthetic_df.head())

# Quick output profile for quality-control only; formal fidelity is measured in Part 10.
profile = pd.DataFrame({
    'column': ALL_COLS,
    'unique_synthetic_categories': [int(synthetic_df[c].nunique()) for c in ALL_COLS],
    'missing_token_rate': [float((synthetic_df[c] == schema.get('missing_token', '__MISSING__')).mean()) for c in ALL_COLS],
    'other_token_rate': [float((synthetic_df[c] == schema.get('other_token', '__OTHER__')).mean()) for c in ALL_COLS],
})
display(profile)
profile.to_csv(METRICS_DIR / f'part9_profile_{MODEL_LABEL}.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, title in [(axes[0], TARGET_COL, 'Synthetic target distribution'), (axes[1], 'GENHLTH', 'Synthetic general-health distribution')]:
    counts = synthetic_df[col].value_counts().reindex(DOMAINS[col], fill_value=0)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel('Synthetic respondents')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
fig.savefig(PLOTS_DIR / f'part9_synthetic_profile_{MODEL_LABEL}.png', dpi=180, bbox_inches='tight')
plt.show()
print('Part 9 complete. Synthetic data is saved in:', OUTPUT_PATH)

# ## 7. Research and privacy notes
#
# - The synthetic data are sampled from the selected model, not copied from an original row by design. Exact-match and distance-based checks are still required and are performed in Part 12.
# - `argmax` ensures exactly one category per field, but it can conceal poor calibration. Part 10 compares the decoded distribution and cross-field associations with real held-out respondents.
# - A file generated from a DP-trained checkpoint inherits the model's DP training guarantee only when the model training mechanism and accountant are valid and correctly reported. Post-processing does not spend additional privacy budget, but releasing multiple separately trained models must be composed.
# - `secure_mode=False` in Part 7 must remain disclosed. This notebook does not convert the public BRFSS source into confidential medical records.
#
# ## References
#
# 1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*. https://arxiv.org/abs/2006.11239
# 2. Opacus documentation. https://opacus.ai/
# 3. CDC. 2025 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2025.html
