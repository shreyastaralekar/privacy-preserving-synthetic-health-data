#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part6_DP_Medical_NonPrivate_Training.ipynb.

Run from the project root with: python scripts/part06_nonprivate_training.py
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
print("Part 06 – Part6_DP_Medical_NonPrivate_Training")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 6 – Non-Private DDPM Training Baseline
#
# **Goal:** Train the same tabular denoiser without DP-SGD to establish a reference generator. This is the non-private comparison model; it has **no formal differential privacy guarantee**.
#
# **Inputs:** `data/encoded_train.npy`, `data/ddpm_config.json` from Part 5.  
# **Outputs:** model checkpoint, epoch-level resume checkpoint, training history CSV and loss plot.
#
# Run this notebook after Part 5. If a Colab runtime is reset, remount Drive and rerun; a completed checkpoint is reused, and an interrupted run can resume from the most recent saved epoch.

# ## 1. Dependencies and persistent project setup

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

# ## 2. Load the saved arrays and architecture configuration

TRAIN_ARRAY_PATH = DATA_DIR / 'encoded_train.npy'
CONFIG_PATH = DATA_DIR / 'ddpm_config.json'
if not TRAIN_ARRAY_PATH.exists() or not CONFIG_PATH.exists():
    raise FileNotFoundError('Run Part 5 first; expected encoded_train.npy and ddpm_config.json in data/.')
with open(CONFIG_PATH, encoding='utf-8') as f:
    config = json.load(f)
X_train_tab = np.load(TRAIN_ARRAY_PATH).astype(np.float32)
INPUT_DIM = int(config['input_dim'])
if X_train_tab.ndim != 2 or X_train_tab.shape[1] != INPUT_DIM:
    raise ValueError('Encoded training array and model configuration dimensions do not match.')
if len(X_train_tab) != int(config['train_rows']):
    raise ValueError('Encoded row count differs from the saved Part 5 manifest.')
TIMESTEPS = int(config['timesteps'])
HIDDEN_DIM = int(config['hidden_dim'])
TIME_EMBED_DIM = int(config['time_embed_dim'])
BATCH_SIZE = int(config['batch_size'])
EPOCHS = int(config['epochs'])
LEARNING_RATE = float(config['learning_rate'])
print('Training rows:', len(X_train_tab), '| input dimension:', INPUT_DIM)
print('Epochs:', EPOCHS, '| batch size:', BATCH_SIZE, '| device:', DEVICE)

# ## 3. Rebuild the identical denoiser and diffusion schedule

import math
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

def unwrap_model(model):
    for attr in ('_module', 'module'):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return model

train_tensor = torch.from_numpy(X_train_tab)
train_loader = DataLoader(
    TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True,
    drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(SEED)
)

# ## 4. Train with ordinary Adam (no privacy protection)
#
# The objective remains per-respondent noise-prediction MSE. Unlike Part 7, this training loop does not clip per-example gradients or add calibrated Gaussian noise.

FINAL_PATH = CHECKPOINT_DIR / 'ddpm_non_private.pt'
RESUME_PATH = CHECKPOINT_DIR / 'ddpm_non_private_resume.pt'
HISTORY_PATH = METRICS_DIR / 'nonprivate_training_history.csv'
FORCE_RETRAIN = False  # Set True only when intentionally starting a fresh non-private run.

model = TabularDenoiser(INPUT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
history = []
start_epoch = 0

if FINAL_PATH.exists() and not FORCE_RETRAIN:
    saved = torch.load(FINAL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)['mean_loss'].astype(float).tolist()
    print('Found completed non-private model; loaded existing checkpoint. Set FORCE_RETRAIN=True to retrain.')
elif RESUME_PATH.exists() and not FORCE_RETRAIN:
    saved = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
    if saved.get('data_fingerprint') != config.get('data_fingerprint'):
        raise ValueError('Resume checkpoint belongs to a different training dataset. Remove it only if a fresh run is intended.')
    model.load_state_dict(saved['model_state_dict'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    history = list(saved.get('history', []))
    start_epoch = int(saved.get('completed_epochs', 0))
    print(f'Resuming from completed epoch {start_epoch}.')
else:
    print('Starting a fresh non-private run.')

if not (FINAL_PATH.exists() and not FORCE_RETRAIN):
    start_time = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        losses = []
        for (x0,) in train_loader:
            x0 = x0.to(DEVICE)
            per_example = diffusion_per_example_loss(model, x0, SCHEDULE, TIMESTEPS)
            loss = per_example.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float('nan')
        history.append(epoch_loss)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'completed_epochs': epoch + 1,
            'history': history,
            'data_fingerprint': config.get('data_fingerprint'),
            'config': config,
        }, RESUME_PATH)
        pd.DataFrame({'epoch': range(1, len(history)+1), 'mean_loss': history}).to_csv(HISTORY_PATH, index=False)
        print(f'Epoch {epoch+1}/{EPOCHS} | mean loss {epoch_loss:.6f} | elapsed {(time.time()-start_time)/60:.1f} min')

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
        'architecture': 'TabularDenoiser',
        'config': config,
        'data_fingerprint': config.get('data_fingerprint'),
        'training_history': history,
        'privacy_status': 'NON_PRIVATE – no formal differential privacy guarantee',
    }, FINAL_PATH)
    if RESUME_PATH.exists():
        RESUME_PATH.unlink()
    print('Saved final checkpoint:', FINAL_PATH)
else:
    print('No training performed because a completed checkpoint was reused.')

if history:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history)+1), history, marker='o')
    ax.set_title('Non-private DDPM training loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean denoising MSE')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'part6_nonprivate_training_loss.png', dpi=180, bbox_inches='tight')
    plt.show()
    pd.DataFrame({'epoch': range(1, len(history)+1), 'mean_loss': history}).to_csv(HISTORY_PATH, index=False)

print('Checkpoint exists:', FINAL_PATH.exists())
print('History:', HISTORY_PATH)

# ## 5. Interpretation and handoff
#
# A decreasing loss indicates that the model is learning to predict the injected Gaussian noise on the training representation. It does **not** establish statistical fidelity, downstream usefulness, or privacy. Those are assessed in later parts. Keep this checkpoint as the non-private comparator; never label it differentially private.
