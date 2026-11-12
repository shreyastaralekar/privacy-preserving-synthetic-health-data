#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part7_DP_Medical_DP_SGD_Training.ipynb.

Run from the project root with: python scripts/part07_dp_sgd_training.py
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
print("Part 07 – Part7_DP_Medical_DP_SGD_Training")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 7 – Differentially Private DDPM Training with Opacus
#
# **Goal:** Train separate respondent-level DP-SGD denoisers for target privacy budgets ε = 1, 3 and 8, using Opacus.
#
# **Inputs:** encoded respondent rows and configuration from Part 5.  
# **Outputs:** private model checkpoints, per-run accountant states, run metadata, training histories and plots.
#
# ### Important training design
#
# - One training example is one BRFSS respondent row.
# - The loss function returns one scalar per respondent before the batch mean is taken.
# - Opacus performs per-example gradient computation, clipping and Gaussian noise addition.
# - Poisson sampling is enabled for the sampling assumption used by the Opacus accountant.
# - The achieved ε is measured from the accountant after training; the requested ε is only a target used to calibrate noise.
# - Each target ε is a separate training run on the same respondent dataset. Releasing multiple trained models composes their privacy losses; Part 8 calculates a basic composition summary.
#
# The dataset is public CDC BRFSS survey microdata, not hospital EHR. This is a DP-method demonstration and does not make the already-public source file confidential.

# ## 1. Install Opacus and prepare the persistent Drive workspace

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

# ## 2. Load encoded rows and the frozen model configuration

from opacus import PrivacyEngine
import opacus

CONFIG_PATH = DATA_DIR / 'ddpm_config.json'
TRAIN_ARRAY_PATH = DATA_DIR / 'encoded_train.npy'
if not CONFIG_PATH.exists() or not TRAIN_ARRAY_PATH.exists():
    raise FileNotFoundError('Run Part 5 first.')
with open(CONFIG_PATH, encoding='utf-8') as f:
    config = json.load(f)
X_train_tab = np.load(TRAIN_ARRAY_PATH).astype(np.float32)
INPUT_DIM = int(config['input_dim'])
if X_train_tab.shape != (int(config['train_rows']), INPUT_DIM):
    raise ValueError('Encoded training array shape does not match the Part 5 manifest.')

TIMESTEPS = int(config['timesteps'])
HIDDEN_DIM = int(config['hidden_dim'])
TIME_EMBED_DIM = int(config['time_embed_dim'])
BATCH_SIZE = int(config['batch_size'])
EPOCHS = int(config['epochs'])
LEARNING_RATE = float(config['learning_rate'])
MAX_GRAD_NORM = float(config['max_grad_norm'])
TARGET_EPSILONS = [float(x) for x in config.get('target_epsilons', [1.0, 3.0, 8.0])]
ACCOUNTANT = str(config.get('accountant', 'prv'))
SECURE_MODE = bool(config.get('secure_mode', False))
N_TRAIN = int(len(X_train_tab))
DELTA = min(1e-5, 1.0 / (10.0 * N_TRAIN))
if not DELTA < 1.0 / N_TRAIN:
    raise ValueError('delta must be strictly smaller than 1/N for the selected respondent-level dataset.')

print('Opacus:', opacus.__version__)
print('N train:', N_TRAIN, '| delta:', DELTA, '| accountant:', ACCOUNTANT)
print('Target epsilons:', TARGET_EPSILONS, '| epochs:', EPOCHS, '| batch:', BATCH_SIZE)
print('secure_mode:', SECURE_MODE, '(False is convenient for Colab but does not provide cryptographic RNG hardening)')

# ## 3. Recreate the identical denoiser and define respondent-level loss

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

def build_base_loader():
    tensor = torch.from_numpy(X_train_tab)
    return DataLoader(
        TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True,
        drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(SEED)
    )

def epsilon_tag(eps):
    return str(float(eps)).replace('.', 'p')

# ## 4. Run the DP-SGD experiments
#
# A progress checkpoint is saved after every completed epoch. When the runtime restarts, the notebook can reconstruct the private optimizer using the already-calibrated noise multiplier, restore the saved accountant history, and continue composing privacy loss from the prior epochs.

FORCE_RETRAIN = False  # Set True to deliberately discard completed/partial checkpoints for each selected epsilon.
RESUME_PARTIAL = True
EPSILONS_TO_RUN = [3.0]  # For a short smoke test, temporarily use [3.0].

privacy_rows = []
all_histories = {}

for target_eps in EPSILONS_TO_RUN:
    tag = epsilon_tag(target_eps)
    final_path = CHECKPOINT_DIR / f'ddpm_dp_eps_{tag}.pt'
    progress_path = CHECKPOINT_DIR / f'ddpm_dp_eps_{tag}_progress.pt'
    accountant_path = CHECKPOINT_DIR / f'accountant_dp_eps_{tag}.pt'
    meta_path = METRICS_DIR / f'dp_run_eps_{tag}.json'
    history_path = METRICS_DIR / f'dp_history_eps_{tag}.csv'

    if final_path.exists() and accountant_path.exists() and meta_path.exists() and not FORCE_RETRAIN:
        with open(meta_path, encoding='utf-8') as f:
            existing_meta = json.load(f)
        if (existing_meta.get('data_fingerprint') != config.get('data_fingerprint')
                or int(existing_meta.get('epochs', -1)) != EPOCHS
                or float(existing_meta.get('target_epsilon', -1)) != float(target_eps)):
            raise ValueError('A completed DP checkpoint exists but its dataset/configuration does not match this run. Set FORCE_RETRAIN=True only if you intend to replace it.')
        print(f"\nSkipping completed ε target {target_eps:g}; saved checkpoint already exists.")
        privacy_rows.append(existing_meta)
        if history_path.exists():
            all_histories[f'ε target {target_eps:g}'] = pd.read_csv(history_path)['mean_loss'].astype(float).tolist()
        continue

    base_model = TabularDenoiser(INPUT_DIM).to(DEVICE)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=LEARNING_RATE)
    base_loader = build_base_loader()
    privacy_engine = PrivacyEngine(accountant=ACCOUNTANT, secure_mode=SECURE_MODE)
    history = []
    start_epoch = 0
    resumed = False

    if progress_path.exists() and RESUME_PARTIAL and not FORCE_RETRAIN:
        progress = torch.load(progress_path, map_location=DEVICE, weights_only=False)
        expected = {
            'target_epsilon': float(target_eps),
            'delta': float(DELTA),
            'epochs': int(EPOCHS),
            'data_fingerprint': config.get('data_fingerprint'),
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise ValueError(f'Partial checkpoint mismatch for {key}. Do not mix runs with different data/configuration.')

        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=base_model,
            optimizer=base_optimizer,
            data_loader=base_loader,
            noise_multiplier=float(progress['noise_multiplier']),
            max_grad_norm=MAX_GRAD_NORM,
            loss_reduction='mean',
            poisson_sampling=True,
            rand_on_empty=True,
        )
        unwrap_model(private_model).load_state_dict(progress['model_state_dict'])
        try:
            private_optimizer.load_state_dict(progress['optimizer_state_dict'])
        except Exception as exc:
            warnings.warn(f'Adam optimizer state could not be restored; continuing with fresh Adam state. DP accountant/model state will still be restored. Details: {exc}')
        privacy_engine.accountant.load_state_dict(progress['accountant_state_dict'])
        history = list(progress.get('loss_history', []))
        start_epoch = int(progress.get('completed_epochs', 0))
        resumed = True
        noise_multiplier = float(progress['noise_multiplier'])
        print(f"\nResuming ε target {target_eps:g} from epoch {start_epoch}; restored accountant state.")
    else:
        private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
            module=base_model,
            optimizer=base_optimizer,
            data_loader=base_loader,
            target_epsilon=float(target_eps),
            target_delta=float(DELTA),
            epochs=int(EPOCHS),
            max_grad_norm=float(MAX_GRAD_NORM),
            loss_reduction='mean',
            poisson_sampling=True,
            rand_on_empty=True,
        )
        noise_multiplier = float(private_optimizer.noise_multiplier)
        print(f"\nStarting ε target {target_eps:g}; calibrated noise multiplier={noise_multiplier:.6f}")

    start_time = time.time()
    for epoch in range(start_epoch, EPOCHS):
        private_model.train()
        losses = []
        for (x0,) in private_loader:
            x0 = x0.to(DEVICE)
            if x0.shape[0] == 0:
                continue
            per_example = diffusion_per_example_loss(private_model, x0, SCHEDULE, TIMESTEPS)
            loss = per_example.mean()
            private_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            private_optimizer.step()
            losses.append(float(loss.detach().cpu()))

        epoch_loss = float(np.mean(losses)) if losses else float('nan')
        history.append(epoch_loss)
        progress_payload = {
            'target_epsilon': float(target_eps),
            'delta': float(DELTA),
            'epochs': int(EPOCHS),
            'completed_epochs': epoch + 1,
            'noise_multiplier': float(noise_multiplier),
            'max_grad_norm': float(MAX_GRAD_NORM),
            'model_state_dict': {k: v.detach().cpu() for k, v in unwrap_model(private_model).state_dict().items()},
            'optimizer_state_dict': private_optimizer.state_dict(),
            'accountant_state_dict': privacy_engine.accountant.state_dict(),
            'loss_history': history,
            'data_fingerprint': config.get('data_fingerprint'),
            'accountant': ACCOUNTANT,
        }
        torch.save(progress_payload, progress_path)
        pd.DataFrame({'epoch': range(1, len(history)+1), 'mean_loss': history}).to_csv(history_path, index=False)
        print(f"ε target {target_eps:g} | epoch {epoch+1}/{EPOCHS} | loss={epoch_loss:.6f} | elapsed={(time.time()-start_time)/60:.1f} min")

    achieved_epsilon = float(privacy_engine.get_epsilon(delta=float(DELTA)))
    accountant_state = privacy_engine.accountant.state_dict()
    torch.save({
        'model_state_dict': {k: v.detach().cpu() for k, v in unwrap_model(private_model).state_dict().items()},
        'input_dim': INPUT_DIM,
        'architecture': 'TabularDenoiser',
        'target_epsilon': float(target_eps),
        'achieved_epsilon': achieved_epsilon,
        'delta': float(DELTA),
        'noise_multiplier': float(noise_multiplier),
        'max_grad_norm': float(MAX_GRAD_NORM),
        'epochs': int(EPOCHS),
        'accountant': ACCOUNTANT,
        'data_fingerprint': config.get('data_fingerprint'),
        'privacy_unit': 'one BRFSS respondent row',
    }, final_path)
    torch.save(accountant_state, accountant_path)

    sample_rate = 1.0 / len(base_loader)
    num_steps = int(sum(int(item[-1]) for item in accountant_state.get('history', [])))
    run_meta = {
        'model': f'dp_eps_{target_eps:g}',
        'target_epsilon': float(target_eps),
        'achieved_epsilon': achieved_epsilon,
        'delta': float(DELTA),
        'noise_multiplier': float(noise_multiplier),
        'max_grad_norm': float(MAX_GRAD_NORM),
        'epochs': int(EPOCHS),
        'respondent_train_n': N_TRAIN,
        'batch_size_nominal': BATCH_SIZE,
        'sample_rate_per_step': float(sample_rate),
        'accountant': ACCOUNTANT,
        'accountant_steps': num_steps,
        'data_fingerprint': config.get('data_fingerprint'),
        'privacy_unit': 'one BRFSS respondent row',
        'secure_mode': SECURE_MODE,
        'resumed_from_partial_checkpoint': bool(resumed),
        'model_checkpoint': str(final_path.name),
        'accountant_checkpoint': str(accountant_path.name),
        'status': 'completed',
    }
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding='utf-8')
    if progress_path.exists():
        progress_path.unlink()
    privacy_rows.append(run_meta)
    all_histories[f'ε target {target_eps:g}'] = history
    print(f"Completed ε target {target_eps:g}: accountant ε={achieved_epsilon:.6f}, δ={DELTA:.2e}")

# Refresh a single project-level table from all completed DP run metadata.
all_meta = []
for path in sorted(METRICS_DIR.glob('dp_run_eps_*.json')):
    with open(path, encoding='utf-8') as f:
        all_meta.append(json.load(f))
if all_meta:
    privacy_df = pd.DataFrame(all_meta).sort_values('target_epsilon')
    privacy_df.to_csv(METRICS_DIR / 'privacy_accounting.csv', index=False)
    display(privacy_df[['model','target_epsilon','achieved_epsilon','delta','noise_multiplier','accountant_steps']])
else:
    print('No completed DP models yet.')

if all_histories:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in all_histories.items():
        ax.plot(range(1, len(values)+1), values, marker='o', label=label)
    ax.set_title('DP-DDPM denoising loss by target privacy budget')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean denoising MSE')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'part7_dp_training_losses.png', dpi=180, bbox_inches='tight')
    plt.show()
print('DP model and accountant checkpoints are saved in:', CHECKPOINT_DIR)

# ## 5. Interpretation and essential privacy cautions
#
# - Smaller target ε generally requires a larger noise multiplier and can make optimisation harder. The measured result is the accountant's achieved ε, not the requested target alone.
# - The non-private model from Part 6 is not covered by this guarantee.
# - `secure_mode=False` is used for compatibility in a free Colab environment; do not claim cryptographically secure noise generation from this setting.
# - Differential privacy applies to the documented respondent-level training mechanism under its stated sampling and adjacency assumptions. It does not automatically protect data copied outside that mechanism.
# - Keep each model's matching accountant checkpoint and metadata together. Part 8 verifies those saved states.
