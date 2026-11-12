# DP Medical Synthetic Project – Local Python IDE Package

This package converts the modular research workflow (Parts 2–14) from Colab notebooks into ordinary `.py` scripts suitable for PyCharm, VS Code, Spyder, or a terminal. It uses local disk storage and does **not** mount Google Drive.

## 1. Recommended computer setup

- Windows 10/11, macOS, or Linux
- Python 3.11 recommended (use a 64-bit installation)
- 16 GB RAM recommended for the 30,000-row pilot configuration
- NVIDIA GPU optional; CPU mode works but diffusion/DP training can be slow
- Keep at least 5–10 GB free for source data, generated data, checkpoints and plots

## 2. Folder layout

```text
DP_Medical_Synthetic_Project_2025/
├── scripts/                 # Run Parts 2–14 in order
├── reference/
│   └── part01_master_reference_do_not_run.py
├── data/                    # LLCP2025.XPT and processed CSV/NPY/JSON
├── plots/                   # EDA, model and evaluation graphs
├── checkpoints/             # Training/resume and accountant checkpoints
├── models/                  # Reserved for exported model files
├── metrics/                 # CSV/JSON/Markdown evaluation results
├── artifacts/synthetic_data/# Generated synthetic CSV + metadata
├── final_deliverables/      # Final DOCX/PPTX/PDF/ZIP
├── logs/
├── project_utils.py
├── run_parts.py
└── requirements.txt
```

## 3. Install PyTorch first

Open a terminal in this project folder. Create and activate a virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### PyTorch CPU or GPU

For a CPU-only installation (simplest to start):

```bash
python -m pip install torch
```

For NVIDIA GPU/CUDA, do **not** guess a CUDA wheel command. Open the official PyTorch selector, choose your operating system, Pip, Python and the CUDA version supported by your installed NVIDIA driver, then run the exact command it gives:

https://pytorch.org/get-started/locally/

Then install the rest of the project requirements:

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import torch, opacus, pandas; print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('Opacus', opacus.__version__); print('pandas', pandas.__version__)"
```

If `CUDA False` appears, scripts use CPU. This does not mean the scripts are broken.

## 4. Add the CDC dataset

Download the official CDC 2025 BRFSS annual data and extract the SAS Transport file. Copy the extracted file here:

```text
data/LLCP2025.XPT
```

CDC source page:
https://www.cdc.gov/brfss/annual_data/annual_2025.html

Do not put only the ZIP into `data/`; Part 2 expects the extracted `.XPT` file. If its filename differs, keep one `.XPT`/`.XPORT` file in `data/` (Part 2 discovers it).

## 5. Open in PyCharm

1. Extract this ZIP to a permanent local folder, for example `D:\Research\DP_Medical_Synthetic_Project_2025`.
2. Open PyCharm → **Open** → select the extracted `DP_Medical_Synthetic_Project_2025` folder.
3. Configure the project interpreter: **Settings → Project → Python Interpreter** → select the `.venv` created above.
4. Set every Run Configuration's **Working directory** to the extracted project root (the folder that contains `run_parts.py`). The scripts also resolve their root from their own file location, but this makes commands and relative paths predictable.
5. Open `scripts/part02_preprocessing.py` and click Run, or use the terminal commands below.

VS Code: open the same folder, select `.venv` using **Python: Select Interpreter**, then run scripts from the integrated terminal.

## 6. Correct execution order

**Part 1 is an original all-in-one prototype and is retained only for reference. Do not execute it along with the modular workflow.** The modular workflow starts at Part 2.

Run the first data-preparation stage:

```bash
python run_parts.py --parts 2 3 4
```

Then run the architecture and non-private baseline:

```bash
python run_parts.py --parts 5 6
```

For an initial short DP experiment, open `scripts/part07_dp_sgd_training.py` and change the setting:

```python
EPSILONS_TO_RUN = [3.0]
```

Run the DP training and privacy accountant:

```bash
python run_parts.py --parts 7 8
```

Then run generation and evaluation. The default selected label is `dp_eps_3p0` in Parts 9–12:

```bash
python run_parts.py --parts 9 10 11 12
```

Finally:

```bash
python run_parts.py --parts 13 14
```

You may run one script independently, for example:

```bash
python scripts/part05_ddpm_architecture.py
```

You may also run selected parts, e.g. `python run_parts.py --parts 10 11 12` after synthetic data already exists.

## 7. Model labels and comparison runs

- Part 9 default: DP model at target epsilon 3.0 (`dp_eps_3p0`).
- Parts 10, 11 and 12 must use the same `MODEL_LABEL` as the synthetic dataset, normally `dp_eps_3p0`.
- For epsilon 1 or 8, first train the corresponding model in Part 7, verify it in Part 8, then set the selected epsilon/model label in Part 9 and repeat Parts 9–12 for that label.
- Do not overwrite/compare artifacts from different dataset fingerprints or changed encoding schemas. If you change the row sample, seed, features, or schema, rerun Parts 2 onward.

## 8. Output locations

All outputs are kept in this extracted project folder:

- `data/`: selected data, train/test CSV, encoded NPY, model config and schema manifest
- `plots/`: PNG figures
- `checkpoints/`: model weights, resumable training state and Opacus accountant states
- `metrics/`: privacy verification, fidelity, utility and audit tables/reports
- `artifacts/synthetic_data/`: synthetic datasets and metadata
- `final_deliverables/`: final Word, PowerPoint, PDF and ZIP (created by Part 14)

Back up the entire project folder to another disk/cloud storage. Local IDE runs do not automatically sync to Google Drive.

## 9. Troubleshooting

- **No XPT file:** place extracted `LLCP2025.XPT` in `data/`.
- **Missing library:** activate `.venv`, then run `python -m pip install -r requirements.txt`.
- **Wrong interpreter in PyCharm:** select the `.venv` interpreter, not system Python.
- **CUDA unavailable:** use CPU or install the correct PyTorch CUDA build from the official selector.
- **Out of memory:** leave `MAX_ROWS = 30_000` (or reduce it), lower batch size in the relevant training script, and close other applications. Changing sample size requires rerunning Parts 2 onward.
- **Part 9 says checkpoint is missing:** complete Part 6 for non-private or Part 7 and Part 8 for DP before Part 9.
- **Part 10/11/12 cannot find synthetic CSV:** complete Part 9 and make `MODEL_LABEL` match the output filename.
- **Achieved epsilon differs from target:** trust the Part 8 accountant result. Do not report the target epsilon as achieved unless verification passes and the target budget check says WITHIN_TARGET.
- **Training stopped:** restart the same script with the same data/config. Part 6/7 include checkpoint/resume logic, but do not change configuration midway through a run.

## 10. Research and privacy cautions

- BRFSS is a public, self-reported health survey, not hospital EHR data.
- This project's DP adjacency unit is intended to be one respondent row in the selected training split.
- The non-private model has no formal DP guarantee.
- `secure_mode=False` is not cryptographically hardened randomness. Do not describe this package as a production-ready secure release.
- Exact-match, DCR and membership inference are empirical diagnostics; they do not replace formal DP accounting.
- Releasing multiple separately trained models on the same respondents incurs privacy composition. Follow Part 8's composed bound when multiple models are released.
- No result is considered complete until the real dataset run has produced and saved the corresponding artifacts.

## 11. Official installation and dataset references

- PyTorch installation selector: https://pytorch.org/get-started/locally/
- Opacus documentation: https://opacus.ai/docs/introduction.html
- CDC BRFSS 2025 annual data: https://www.cdc.gov/brfss/annual_data/annual_2025.html
