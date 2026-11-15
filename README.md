# Privacy-Preserving Synthetic Health Data Generation Using Diffusion Models

**A tabular Gaussian DDPM and Differentially Private SGD study using the CDC 2025 BRFSS public dataset**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Privacy](https://img.shields.io/badge/Privacy-DP--SGD%20%2B%20Opacus-green.svg)](https://opacus.ai/)
[![Status](https://img.shields.io/badge/Research%20status-Experimental%20pipeline-lightgrey.svg)](#current-research-status)

**GitHub repository:** <https://github.com/shreyastaralekar/privacy-preserving-synthetic-health-data>

> **Research-status notice:** This repository contains the implementation and evaluation pipeline. Do not treat configured privacy budgets as achieved results, and do not report numerical fidelity, utility, or privacy findings until the corresponding experiments have completed and their output files have been checked. Replace this notice with a results summary only after verifying the actual run artefacts.

---

## 1. Project Overview and Abstract

Health-survey datasets are useful for public-health analysis, statistical modelling, and the development of research software. However, sharing respondent-level records can raise privacy concerns. Synthetic data offers a possible way to create artificial records that preserve selected statistical properties of a source dataset. Synthetic data is not automatically private: a generator can overfit or retain information about training records. This project therefore studies synthetic tabular data generation together with a formal privacy mechanism applied during model training.

The project implements a lightweight **Gaussian Denoising Diffusion Probabilistic Model (DDPM)** over one-hot encoded categorical health-survey records. A non-private model is used as a reference, and differentially private variants are trained using **Differentially Private Stochastic Gradient Descent (DP-SGD)** through Opacus. The intended privacy-budget configurations are ε = 1, 3, and 8; the achieved ε must be taken from the privacy accountant for each completed run. Synthetic records are evaluated using marginal distribution distance, pairwise categorical association, downstream prediction utility, and selected empirical disclosure-risk diagnostics.

The benchmark is the **2025 Behavioral Risk Factor Surveillance System (BRFSS)** public annual combined landline and cellular dataset released by the U.S. Centers for Disease Control and Prevention (CDC). BRFSS is a cross-sectional, self-reported telephone survey; it is **not hospital electronic health-record (EHR) data**. The project is a research prototype and is not a clinical decision system or a production privacy certification.

## 2. Problem Statement and Research Objectives

### 2.1 Problem Statement

A tabular generator trained without privacy constraints may learn useful population-level patterns but can potentially memorise characteristics of individual training rows. DP-SGD limits the contribution of each training record by clipping per-example gradients and adding calibrated noise. The noise needed for privacy can affect the quality and usefulness of the generated data. The project compares a non-private DDPM with DP-trained variants under a common data schema and evaluation protocol to study this privacy–utility relationship.

### 2.2 Aim

To design, implement, and evaluate a reproducible privacy-aware pipeline for synthetic health-survey data generation using a tabular Gaussian DDPM trained with and without respondent-level DP-SGD.

### 2.3 Objectives

1. Obtain and validate the CDC 2025 BRFSS annual SAS Transport dataset.
2. Select 23 demographic and health-related predictor variables and the categorical `DIABETE4` field.
3. Preserve documented category codes, missing values, and out-of-domain values using a fixed category schema.
4. Create a reproducible respondent-level training and held-out testing split.
5. Establish a real-data baseline classifier for the clear Yes/No diabetes responses.
6. Implement and train a Gaussian DDPM on concatenated one-hot categorical vectors.
7. Train a non-private reference model and DP-SGD variants configured for target ε values of 1, 3, and 8.
8. Record achieved ε and δ from the Opacus privacy accountant, together with training configuration and software versions.
9. Generate and decode synthetic records using the saved model and fixed schema.
10. Evaluate statistical fidelity, downstream utility, and selected empirical privacy risks.
11. Compare the results across privacy budgets and document limitations and safe-use boundaries.

### 2.4 Research Questions

- **RQ1 – Fidelity:** How closely do synthetic records reproduce real marginal distributions and pairwise categorical associations?
- **RQ2 – Utility:** Can a classifier trained on synthetic data retain predictive performance on held-out real respondents?
- **RQ3 – Privacy:** What formal privacy budget is reported by the accountant, and what do the selected empirical diagnostics indicate?
- **RQ4 – Trade-off:** How do fidelity and predictive utility vary with the achieved privacy budget?

## 3. Dataset Source and Acquisition Procedure

### 3.1 Dataset

The project uses the **CDC 2025 BRFSS annual combined landline and cellular SAS Transport file**. The CDC's 2025 annual data page reports 356,158 records and 284 variables in the combined SAS Transport file. The annual data page also provides the codebook, overview, calculated-variable documentation, data-quality report, and survey-weighting guidance [1], [2].

- Official CDC data page: <https://www.cdc.gov/brfss/annual_data/annual_2025.html>
- Dataset file expected by the preprocessing script: `LLCP2025.XPT`
- File format: SAS Transport (`.XPT`)
- Source type: public, cross-sectional, self-reported health survey

### 3.2 Download and Place the Dataset

1. Open the official CDC 2025 BRFSS annual data page: <https://www.cdc.gov/brfss/annual_data/annual_2025.html>.
2. Download the **2025 BRFSS Data (SAS Transport Format)** archive.
3. Extract the archive on your computer.
4. Copy the extracted `.XPT` file into the repository's `data/` directory.
5. The expected path is:

```text
privacy-preserving-synthetic-health-data/
└── data/
    └── LLCP2025.XPT
```

The preprocessing script discovers a single `.XPT`, `.xpt`, or `.xport` file in `data/`. Keep only the intended 2025 annual file in that directory when running Part 2. **The dataset is not included in this Git repository.** Download it from the official CDC source and follow the CDC terms and your institution's data-handling requirements.

### 3.3 Selected Variables

The model schema contains 24 categorical columns: 23 predictors and `DIABETE4`.

| Domain | Variables |
|---|---|
| Demographic / socioeconomic | `SEXVAR`, `MARITAL`, `EDUCA`, `EMPLOY1`, `INCOME3`, `_AGE_G` |
| General health / healthy days | `GENHLTH`, `PHYSHLTH`, `MENTHLTH` |
| Activity / lifestyle | `EXERANY2`, `_TOTINDA`, `_SMOKER3`, `_BMI5CAT` |
| Chronic conditions | `BPHIGH6`, `TOLDHI3`, `CVDINFR4`, `CVDCRHD4`, `CVDSTRK3`, `ASTHMA3`, `CHCCOPD3`, `ADDEPEV3`, `CHCKDNY2`, `HAVARTH4` |
| Generation target | `DIABETE4` |

For the generator, the documented `DIABETE4` response categories are retained in the fixed schema. For the separate binary classification task, code `1` is treated as **Yes** and code `3` as **No**; other responses are not silently re-labelled as either class. Confirm exact response wording and valid code domains against the 2025 CDC codebook before interpreting results.

### 3.4 Preprocessing Defaults

| Setting | Current implementation default |
|---|---:|
| Maximum sample size | 30,000 rows (optional cap) |
| Random seed | 42 |
| Train/test split | 80% / 20% |
| Predictor variables | 23 |
| Generation target | `DIABETE4` |
| Model input | Concatenated one-hot categorical blocks |

If 30,000 valid rows are selected, the nominal split is 24,000 training rows and 6,000 test rows. The actual counts must be read from `preprocessing_metadata.json`; do not assume those counts if the source file or filters differ.

**Survey-design limitation:** The current generator does not incorporate BRFSS survey weights or design-based variance estimation. Its synthetic proportions describe the unweighted analytic sample used by the model and must not be presented as nationally weighted prevalence estimates [2], [3].

## 4. DDPM and DP-SGD Architecture

### 4.1 Overall Workflow

```text
CDC 2025 BRFSS XPT file
          |
          v
Schema validation and selection of 23 predictors + DIABETE4
          |
          v
Reproducible sample and respondent-level train/test split
          |
          v
Fixed categorical domains and one-hot encoding
          |
          v
Gaussian tabular DDPM (time-conditioned MLP denoiser)
          |
          +-----------------------------+
          |                             |
          v                             v
Non-private training              Opacus DP-SGD training
(no formal DP guarantee)          (target epsilon: 1, 3, 8)
          |                             |
          +--------------+--------------+
                         |
                         v
               Reverse diffusion sampling
                         |
                         v
              Decode fixed categorical schema
                         |
                         v
       Fidelity / TSTR-TRTR / privacy-risk diagnostics
                         |
                         v
        Privacy–utility comparison and final reporting
```

### 4.2 Gaussian Tabular DDPM

The model uses a time-conditioned multilayer perceptron to predict the Gaussian noise added to a one-hot encoded row. The forward diffusion process gradually corrupts an encoded row, and the reverse process learns to denoise samples from a Gaussian prior. The implementation uses block-wise argmax decoding to map generated vectors back to categorical values.

The current prototype defaults are:

| Parameter | Default |
|---|---:|
| Diffusion steps | 100 |
| Denoiser hidden dimension | 128 |
| Time-embedding dimension | 32 |
| Batch size | 256 |
| Training epochs | 5 |
| Learning rate | 0.0002 |
| Maximum gradient norm | 1.0 |
| Requested target ε values | 1, 3, 8 |
| Accountant | `prv` |
| `secure_mode` | `False` |

These are pilot configuration values, not validated optimal hyperparameters. The exact category schema and encoded dimension are saved in the Part 5 artefacts and should be read from the run manifest rather than assumed.

**Architecture distinction:** This is a lightweight Gaussian DDPM applied to concatenated one-hot tabular vectors. It is inspired by tabular diffusion research, but it is not an exact implementation of TabDDPM or TabDiff [7], [8]. Gaussian noise does not naturally preserve categorical simplex constraints, and argmax decoding does not guarantee realistic combinations across columns.

### 4.3 Differentially Private SGD

The private training path uses Opacus to apply DP-SGD. At a high level, DP-SGD computes per-example gradients, clips their L2 norms, aggregates the clipped gradients, and adds calibrated Gaussian noise before the optimiser update [17], [21]. The Opacus privacy accountant tracks privacy loss over the training steps.

The requested target budgets are ε = 1, 3, and 8. **A target ε is not an achieved result.** Record the achieved epsilon and delta returned by the accountant for each completed run. Also preserve the accountant type, noise multiplier, sample rate, batch size, number of optimiser steps, clipping norm, library versions, and run metadata.

The intended privacy unit is one respondent row in the training partition. The implementation currently uses `secure_mode=False`; this is a research convenience setting and does not provide cryptographic RNG hardening. Do not describe the current code as a production-ready privacy deployment. If multiple models trained on the same respondents are released, their privacy losses compose; do not describe their joint release using only one model's epsilon.

## 5. Project Directory Structure

The local Python project is organised as follows. Some output folders are created automatically on first execution.

```text
privacy-preserving-synthetic-health-data/
│
├── README.md
├── README_LOCAL_IDE.md
├── NOTEBOOK_MAPPING.md
├── CONVERSION_NOTES.md
├── requirements.txt
├── .gitignore
├── check_environment.py
├── project_utils.py
├── run_parts.py
│
├── reference/
│   └── part01_master_reference_do_not_run.py
│
├── scripts/
│   ├── part02_preprocessing.py
│   ├── part03_eda.py
│   ├── part04_baseline_ml.py
│   ├── part05_ddpm_architecture.py
│   ├── part06_nonprivate_training.py
│   ├── part07_dp_sgd_training.py
│   ├── part08_privacy_accounting.py
│   ├── part09_synthetic_generation.py
│   ├── part10_statistical_fidelity.py
│   ├── part11_tstr_trtr_utility.py
│   ├── part12_privacy_audits.py
│   ├── part13_utility_privacy_tradeoff.py
│   └── part14_final_report_builder.py
│
├── data/                  # Source XPT and processed data; do not commit raw records
├── plots/                 # Generated figures
├── checkpoints/           # Training and accountant checkpoints
├── models/                # Exported model files, if any
├── metrics/               # CSV, JSON and Markdown metrics/reports
├── artifacts/
│   ├── synthetic_data/    # Generated synthetic datasets
│   └── tradeoff_analysis/ # Part 13 consolidated trade-off artefacts
├── final_deliverables/    # Part 14 report, slides and package
└── logs/                  # Execution logs, if generated
```

### Important files

| File / directory | Responsibility |
|---|---|
| `README.md` | Main project description and reproducibility guide |
| `README_LOCAL_IDE.md` | Detailed local IDE setup guide |
| `requirements.txt` | Python package dependencies (PyTorch is installed separately) |
| `check_environment.py` | Checks important dependencies and dataset placement |
| `project_utils.py` | Shared project-root, folder and console-display helpers |
| `run_parts.py` | Runs selected modular scripts in numerical order |
| `reference/` | Original integrated Part 1 prototype, reference only |
| `scripts/` | Runnable Parts 2–14 |
| `data/` | Original XPT, processed CSVs, encoded arrays and schema/config files |
| `checkpoints/` | Non-private and DP model checkpoints and accountant state files |
| `metrics/` | Preprocessing, training, privacy-accounting and evaluation results |
| `artifacts/` | Synthetic datasets and integrated trade-off results |
| `final_deliverables/` | Final report, presentation, manifest and ZIP package |

## 6. Python 3.11 Installation Instructions

### 6.1 Requirements

- Python 3.11 (64-bit recommended)
- Windows 10/11, macOS, or Linux
- 16 GB RAM recommended for the default 30,000-row pilot configuration
- NVIDIA GPU optional; CPU execution is supported but model training may be slower
- Approximately 5–10 GB of free disk space recommended for data, checkpoints and outputs

### 6.2 Clone the Repository

```bash
git clone https://github.com/shreyastaralekar/privacy-preserving-synthetic-health-data.git
cd privacy-preserving-synthetic-health-data
```

### 6.3 Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If the `py` launcher is installed, `py -3.11 -m venv .venv` can also be used. If PowerShell reports that `py` is not recognised, use the `python` command shown above and confirm that `python --version` reports Python 3.11.

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 6.4 Install PyTorch

PyTorch is not pinned in the project's current `requirements.txt` because the correct installation command depends on the operating system and whether a CPU or CUDA-enabled build is required.

For a basic CPU setup, run:

```bash
python -m pip install torch
```

For NVIDIA GPU/CUDA, use the official PyTorch installation selector and run the command it generates for your system: <https://pytorch.org/get-started/locally/>.

### 6.5 Install Project Dependencies

```bash
python -m pip install -r requirements.txt
```

**Additional dependency for Parts 13 and 14:** the current scripts import `IPython.display`. The current `requirements.txt` does not list IPython, so install it explicitly:

```bash
python -m pip install ipython
```

For future repository maintenance, add `ipython>=8.0` to `requirements.txt` so a fresh environment can install all script dependencies in one command. After updating that file, the separate IPython installation command above will no longer be necessary.

### 6.6 Verify the Environment

```bash
python check_environment.py
python -c "import torch, opacus, pandas; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Opacus:', opacus.__version__); print('pandas:', pandas.__version__)"
```

If `CUDA available: False` is displayed, PyTorch is using CPU. This is not by itself an execution error.

## 7. Dependency List

The current `requirements.txt` contains:

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays and vector operations |
| `pandas` | Dataset loading, selection and tabular processing |
| `matplotlib` | Graphs and visualisations |
| `seaborn` | Statistical visualisation |
| `scipy` | Statistical distances and tests |
| `scikit-learn` | Baseline and downstream classifiers, metrics and preprocessing utilities |
| `opacus` | DP-SGD wrappers and privacy accounting |
| `tqdm` | Progress indicators |
| `requests` | HTTP utility support |
| `python-docx` | Word report generation in Part 14 |
| `python-pptx` | PowerPoint generation in Part 14 |
| `torch` | DDPM neural network and training; install separately as described above |
| `ipython` | `IPython.display` used by Parts 13 and 14; install separately or add to `requirements.txt` |

The versions actually used in a completed experiment should be recorded with the run metadata. The dependency declarations are minimum/range specifications, not a fully locked environment. For exact reproducibility, generate and preserve a tested lock file after the final environment has been validated.

## 8. Steps to Execute the Project

### 8.1 Important Execution Rule

**Part 1 is an original all-in-one master prototype kept for reference. Do not execute Part 1 together with the modular workflow.** The modular runner supports Parts 2–14 and deliberately does not include Part 1.

Run all commands from the repository root (the folder containing `run_parts.py`). Make sure the virtual environment is active and the dataset is in `data/` before starting.

### 8.2 Execute in Stages

**Stage 1 – Data preparation and baseline**

```bash
python run_parts.py --parts 2 3 4
```

**Stage 2 – DDPM architecture and non-private training**

```bash
python run_parts.py --parts 5 6
```

**Stage 3 – DP-SGD training and privacy accounting**

For an initial pilot, open `scripts/part07_dp_sgd_training.py` and set:

```python
EPSILONS_TO_RUN = [3.0]
```

Then run:

```bash
python run_parts.py --parts 7 8
```

After the ε = 3 pilot completes and Part 8 verifies the accountant output, additional models can be trained by configuring the desired target epsilon values. Do not treat the requested target as an achieved privacy guarantee.

**Stage 4 – Synthetic generation and evaluation**

```bash
python run_parts.py --parts 9 10 11 12
```

Parts 9–12 must use matching model labels. The current default label is `dp_eps_3p0`. If another model is evaluated, update the `MODEL_LABEL` settings in Parts 10, 11 and 12 to match the synthetic dataset generated in Part 9.

**Stage 5 – Integrated privacy–utility analysis and final deliverables**

```bash
python run_parts.py --parts 13 14
```

Parts 13 and 14 require outputs from the previous stages. Part 14 generates its report from available project artefacts and marks missing experimental evidence as pending rather than inventing values.

### 8.3 Run an Individual Part

A single script can also be run directly from the repository root. For example:

```bash
python scripts/part05_ddpm_architecture.py
```

The runner stops when a part returns a non-zero exit code. Resolve the error and rerun the failed part before moving to later stages.

## 9. Expected Outputs from Parts 2–14

The following are the principal output artefacts created by the current local Python scripts. Exact files depend on completed runs and selected model labels.

| Part | Script | Main expected outputs |
|---|---|---|
| 2 – Preprocessing | `part02_preprocessing.py` | `data/selected_data.csv`, `data/train.csv`, `data/test.csv`, `preprocessing_quality.csv`, `preprocessing_metadata.json` |
| 3 – EDA | `part03_eda.py` | `eda_data_quality.csv`, `eda_cramers_v.csv`, `eda_summary.json`; plots such as missingness, diabetes response, feature distributions, healthy days and Cramér's V |
| 4 – Baseline ML | `part04_baseline_ml.py` | `baseline_ml_results.csv`, `baseline_ml_predictions.csv`, `baseline_ml_summary.json`, metric and confusion-matrix plots |
| 5 – DDPM architecture | `part05_ddpm_architecture.py` | `data/encoded_train.npy`, `data/encoded_test.npy`, `data/schema_manifest.json`, `data/ddpm_config.json`, noise-schedule and architecture plots |
| 6 – Non-private training | `part06_nonprivate_training.py` | `checkpoints/ddpm_non_private.pt`, resume checkpoint, `metrics/nonprivate_training_history.csv`, training-loss plot |
| 7 – DP-SGD training | `part07_dp_sgd_training.py` | DP model checkpoints, progress checkpoints, accountant state files, `metrics/dp_run_eps_*.json`, DP training histories and privacy summary |
| 8 – Privacy accounting | `part08_privacy_accounting.py` | `metrics/privacy_accounting_verification.csv`, `metrics/dp_composition_summary.json`, `metrics/privacy_accounting_report.md`, verified-budget plot |
| 9 – Synthetic generation | `part09_synthetic_generation.py` | `artifacts/synthetic_data/synthetic_<model_label>.csv`, metadata JSON, profile CSV and plot |
| 10 – Statistical fidelity | `part10_statistical_fidelity.py` | Per-column marginal fidelity CSV, pairwise association fidelity CSV, summary JSON, real/synthetic association matrices and plots |
| 11 – TSTR/TRTR utility | `part11_tstr_trtr_utility.py` | TSTR/TRTR metrics CSV, details JSON, comparison and ROC plots |
| 12 – Privacy audits | `part12_privacy_audits.py` | Privacy-audit summary JSON, DCR sample CSV, membership-attack sample CSV, audit report Markdown and diagnostic plots |
| 13 – Utility–privacy trade-off | `part13_utility_privacy_tradeoff.py` | `artifacts/tradeoff_analysis/part13_tradeoff_comparison.csv`, summary JSON, Markdown report and trade-off/completeness plots |
| 14 – Final report builder | `part14_final_report_builder.py` | Final DOCX report, PPTX presentation, `final_project_manifest.json`, final deliverables ZIP; PDF is optional when a suitable converter is available |

These are expected output names from the current scripts, not evidence that every output already exists. Inspect the files and run metadata after execution.

## 10. Evaluation Metrics and Privacy Limitations

### 10.1 Statistical Fidelity

- **Jensen–Shannon distance (JSD):** compares the marginal category distributions of real and synthetic data. Values closer to zero indicate more similar marginal distributions. Confirm whether the implementation reports distance or divergence.
- **Cramér's V difference:** compares pairwise categorical association strengths between real and synthetic records. Smaller absolute differences indicate closer association values under the chosen calculation.
- **Category and sentinel diagnostics:** inspect missing, other, rare, and unexpected category frequencies after decoding.

### 10.2 Downstream Predictive Utility

- **TRTR (Train on Real, Test on Real):** trains the evaluation classifier on eligible real training records and tests it on the held-out real test set.
- **TSTR (Train on Synthetic, Test on Real):** trains on synthetic records and tests on the same eligible real held-out test set.
- Report accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. Use the same test partition and label rules for all comparisons.
- For the binary `DIABETE4` task, code 1 is Yes and code 3 is No. Other response codes are excluded from this binary classification task only.

### 10.3 Empirical Privacy Diagnostics

- **Membership-inference AUC:** measures the performance of the particular implemented attack at distinguishing training members from non-members.
- **Distance to closest record (DCR):** examines nearest-record distances from synthetic rows to sampled training and test records under the selected encoded representation.
- **Exact-match rate:** checks whether a complete synthetic categorical row exactly matches a row in the reference data.
- **Synthetic-versus-real distinguishability:** assesses whether a classifier can distinguish the two datasets under the chosen feature representation.

These are empirical diagnostics for specific attacks and samples. They do not replace a formal differential-privacy guarantee and a near-random attack result does not prove that all attacks fail.

### 10.4 Formal Privacy Reporting

For each DP model, report:

- Achieved epsilon (ε) returned by the accountant
- Delta (δ) used in the same run
- Privacy unit and adjacency definition
- Accountant type
- Noise multiplier
- Maximum gradient norm
- Sampling rate / batch size
- Number of optimiser steps and epochs
- Opacus and PyTorch versions
- Whether the run passed the configured target-budget verification

The current implementation configures `secure_mode=False`. This is not cryptographic RNG hardening. The non-private reference model has **no formal DP guarantee**. Releasing multiple models trained on the same respondent dataset incurs privacy composition. See Part 8's accountant output and composition summary before making any release claim.

### 10.5 Dataset and Model Limitations

1. BRFSS is public, self-reported, cross-sectional survey data, not confidential hospital EHR data [1], [3].
2. The default pipeline uses a sample cap of up to 30,000 rows and a single seeded split; actual counts must come from the run metadata.
3. The current model applies Gaussian diffusion to concatenated one-hot categorical data. It is a simplified tabular DDPM baseline, not native mixed-type diffusion.
4. Block-wise argmax decoding ensures a category is chosen for each feature block but does not guarantee realistic cross-column combinations.
5. The current workflow does not apply BRFSS survey weights and therefore does not estimate nationally weighted prevalence [2], [3].
6. The current secure-randomness setting is `False`; this prototype is not a production privacy deployment.
7. Empirical privacy audits cover selected attack settings only.
8. No synthetic output should be used for diagnosis, treatment, triage, or decisions about individuals.
9. No quantitative result should be claimed until it is produced by a completed run and checked against its corresponding output artefact.

## 11. Team, Guide and Institutional Affiliation

**Project team**

- Shreyas Taralekar
- Shantanoo Karad
- Sanket Shitole

**Project Guide:** Prof. Aniket Shahade

**Department:** Artificial Intelligence and Machine Learning

**Institution:** Symbiosis Institute of Technology, Pune Campus, Symbiosis International (Deemed University), Pune, India.

## 12. Academic and Technical References

The following references support the methods, dataset, evaluation approach and implementation tools discussed in this repository. They are listed in IEEE-style numbered format to remain consistent with the project manuscript.

1. Centers for Disease Control and Prevention, “2025 BRFSS Survey Data and Documentation,” CDC, 2026. [Online]. Available: <https://www.cdc.gov/brfss/annual_data/annual_2025.html>.
2. Centers for Disease Control and Prevention, “Complex Sampling Weights and Preparing 2025 BRFSS Module Data for Analysis,” CDC, 2026. [Online]. Available: <https://www.cdc.gov/brfss/annual_data/2025/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2025-508.pdf>.
3. Centers for Disease Control and Prevention, “About the Behavioral Risk Factor Surveillance System (BRFSS),” CDC. [Online]. Available: <https://www.cdc.gov/brfss/about/>.
4. J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020. [Online]. Available: <https://arxiv.org/abs/2006.11239>.
5. Y. Song and S. Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Online]. Available: <https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html>.
6. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-Based Generative Modeling through Stochastic Differential Equations,” in *Proc. International Conference on Learning Representations*, 2021. [Online]. Available: <https://research.google/pubs/score-based-generative-modeling-through-stochastic-differential-equations/>.
7. A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko, “TabDDPM: Modelling Tabular Data with Diffusion Models,” in *Proc. 40th International Conference on Machine Learning*, PMLR, vol. 202, pp. 17564–17579, 2023. [Online]. Available: <https://proceedings.mlr.press/v202/kotelnikov23a.html>.
8. J. Shi, M. Xu, H. Hua, H. Zhang, S. Ermon, and J. Leskovec, “TabDiff: A Mixed-type Diffusion Model for Tabular Data Generation,” in *Proc. International Conference on Learning Representations*, 2025. [Online]. Available: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html>.
9. L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni, “Modeling Tabular Data using Conditional GAN,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Online]. Available: <https://proceedings.neurips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html>.
10. E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun, “Generating Multi-label Discrete Patient Records using Generative Adversarial Networks,” in *Proc. Machine Learning for Healthcare*, PMLR, vol. 68, pp. 286–305, 2017. [Online]. Available: <https://proceedings.mlr.press/v68/choi17a.html>.
11. J. Jordon, J. Yoon, and M. van der Schaar, “PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,” in *Proc. International Conference on Learning Representations*, 2019. [Online]. Available: <https://mlanthology.org/iclr/2019/jordon2019iclr-pategan/>.
12. L. Xie, K. Lin, S. Wang, F. Wang, and J. Zhou, “Differentially Private Generative Adversarial Network,” arXiv:1802.06739, 2018. [Online]. Available: <https://arxiv.org/abs/1802.06739>.
13. J. Zhang, G. Cormode, C. M. Procopiuc, D. Srivastava, and X. Xiao, “PrivBayes: Private Data Release via Bayesian Networks,” in *Proc. ACM SIGMOD International Conference on Management of Data*, pp. 1423–1434, 2014, doi: 10.1145/2588555.2588573.
14. H. Ping, J. Stoyanovich, and B. Howe, “DataSynthesizer: Privacy-Preserving Synthetic Datasets,” in *Proc. Scientific and Statistical Database Management*, 2017, doi: 10.1145/3085504.3091117.
15. C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating Noise to Sensitivity in Private Data Analysis,” in *Theory of Cryptography Conference*, pp. 265–284, 2006.
16. C. Dwork and A. Roth, “The Algorithmic Foundations of Differential Privacy,” *Foundations and Trends in Theoretical Computer Science*, vol. 9, nos. 3–4, pp. 211–487, 2014, doi: 10.1561/0400000042.
17. M. Abadi et al., “Deep Learning with Differential Privacy,” in *Proc. ACM SIGSAC Conference on Computer and Communications Security*, pp. 308–318, 2016. [Online]. Available: <https://arxiv.org/abs/1607.00133>.
18. I. Mironov, “Rényi Differential Privacy,” in *Proc. IEEE 30th Computer Security Foundations Symposium*, pp. 263–275, 2017, doi: 10.1109/CSF.2017.11.
19. Y.-X. Wang, B. Balle, and S. P. Kasiviswanathan, “Subsampled Rényi Differential Privacy and Analytical Moments Accountant,” in *Proc. 22nd International Conference on Artificial Intelligence and Statistics*, PMLR, vol. 89, pp. 1226–1235, 2019. [Online]. Available: <https://proceedings.mlr.press/v89/wang19b.html>.
20. J. Near, D. Darais, N. Lefkovitz, and G. Howarth, “Guidelines for Evaluating Differential Privacy Guarantees,” NIST Special Publication 800-226, Mar. 2025, doi: 10.6028/NIST.SP.800-226.
21. A. Yousefpour, I. Shilov, D. Testuggine, et al., “Opacus: User-Friendly Differential Privacy Library in PyTorch,” *Privacy in Machine Learning Workshop*, NeurIPS, 2021. [Online]. Available: <https://ai.meta.com/research/publications/opacus-user-friendly-differential-privacy-library-in-pytorch/>.
22. R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership Inference Attacks Against Machine Learning Models,” in *Proc. IEEE Symposium on Security and Privacy*, pp. 3–18, 2017, doi: 10.1109/SP.2017.41.
23. S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha, “Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting,” in *Proc. IEEE 31st Computer Security Foundations Symposium*, 2018, doi: 10.1109/CSF.2018.00027.
24. N. Carlini, C. Liu, U. Erlingsson, J. Kos, and D. Song, “The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks,” in *Proc. 28th USENIX Security Symposium*, pp. 267–284, 2019. [Online]. Available: <https://www.usenix.org/conference/usenixsecurity19/presentation/carlini>.
25. J. Lin, “Divergence Measures Based on the Shannon Entropy,” *IEEE Transactions on Information Theory*, vol. 37, no. 1, pp. 145–151, 1991, doi: 10.1109/18.61115.
26. W. Bergsma, “A Bias-Correction for Cramér’s V and Tschuprow’s T,” *Journal of the Korean Statistical Society*, vol. 42, no. 3, pp. 323–328, 2013, doi: 10.1016/j.jkss.2012.10.002.
27. J. P. Reiter, “Releasing Multiply Imputed, Synthetic Public Use Microdata: An Illustration and Empirical Study,” *Journal of the Royal Statistical Society: Series A*, vol. 168, no. 1, pp. 185–205, 2005, doi: 10.1111/j.1467-985X.2004.00343.x.
28. T. E. Raghunathan, J. P. Reiter, and D. B. Rubin, “Multiple Imputation for Statistical Disclosure Limitation,” *Journal of Official Statistics*, vol. 19, no. 1, pp. 1–16, 2003.
29. M. Hernandez, G. Epelde, A. Alberdi, R. Cilla, and D. Rankin, “Synthetic Data Generation for Tabular Health Records: A Systematic Review,” *Neurocomputing*, vol. 493, pp. 28–45, 2022, doi: 10.1016/j.neucom.2022.04.053.
30. H. Murtaza, M. Ahmed, N. F. Khan, G. Murtaza, S. Zafar, and A. Bano, “Synthetic Data Generation: State of the Art in Health Care Domain,” *Computer Science Review*, vol. 48, Art. no. 100546, 2023, doi: 10.1016/j.cosrev.2023.100546.
31. V. C. Pezoulas et al., “Synthetic Data Generation Methods in Healthcare: A Review on Open-Source Tools and Methods,” *Computational and Structural Biotechnology Journal*, vol. 23, pp. 2892–2910, 2024, doi: 10.1016/j.csbj.2024.07.005.
32. Z. Qian, R. Davis, and M. van der Schaar, “Synthcity: A Benchmark Framework for Diverse Use Cases of Tabular Synthetic Data,” in *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, 2023. [Online]. Available: <https://proceedings.neurips.cc/paper_files/paper/2023/file/09723c9f291f6056fd1885081859c186-Paper-Datasets_and_Benchmarks.pdf>.
33. SDV Developers, “SDMetrics: Quality Metrics for Synthetic Data,” documentation. [Online]. Available: <https://docs.sdv.dev/sdmetrics/data-metrics/quality>.
34. Centers for Disease Control and Prevention, “Behavioral Risk Factor Surveillance System (BRFSS): Frequently Asked Questions,” CDC. [Online]. Available: <https://www.cdc.gov/brfss/about/brfss_faq.htm>.
35. A. Yousefpour et al., “Opacus API Reference: PrivacyEngine, DP Optimizers and Privacy Accountants,” Opacus documentation. [Online]. Available: <https://opacus.ai/api/>.

### Additional Software Documentation

- PyTorch installation selector: <https://pytorch.org/get-started/locally/>
- Opacus documentation: <https://opacus.ai/docs/introduction.html>
- Python 3.11 downloads: <https://www.python.org/downloads/release/python-3119/>
- Git documentation: <https://git-scm.com/doc>

---

## Current Research Status

This repository should be described according to the artefacts that have actually been generated on the research machine. The code and workflow configuration alone do not establish completed experimental results.

Before finalising the report, update this table from saved run outputs:

| Evidence | Status / value to enter after execution |
|---|---|
| Actual source rows and rows used | To be filled from `preprocessing_metadata.json` |
| Training and testing rows | To be filled from `preprocessing_metadata.json` |
| Baseline classifier metrics | To be filled from Part 4 outputs |
| Non-private DDPM training status | To be filled from Part 6 checkpoint and history |
| Achieved ε and δ for each DP model | To be filled from Part 8 verification output |
| Synthetic row count and model label | To be filled from Part 9 metadata |
| Fidelity metrics | To be filled from Part 10 outputs |
| TSTR/TRTR metrics | To be filled from Part 11 outputs |
| Privacy audit results | To be filled from Part 12 outputs |
| Integrated privacy–utility comparison | To be filled from Part 13 outputs |

Do not publish made-up values or mark a stage complete merely because its script exists. Preserve the exact configuration, run log, output paths and software versions for every reported experiment.

## Responsible Use and Data Handling

- Do not commit private keys, passwords, access tokens, `.env` files, or personal credentials.
- Do not commit the raw BRFSS file or record-level processed datasets unless there is a documented permission and institutional reason to do so.
- Review generated synthetic data and model checkpoints before publishing them; synthetic data and trained models can still carry disclosure risk.
- Add appropriate ignore patterns to `.gitignore` for local data, virtual environments, checkpoints, models, logs and generated respondent-level artefacts.
- Do not claim that the public BRFSS source has been made confidential by this project.
- Do not claim a production-ready privacy guarantee from the current prototype configuration.
- Do not use generated records for clinical diagnosis or individual decision-making.

## Authors and Affiliation

**Project team**

- Shreyas Taralekar
- Shantanoo Karad
- Sanket Shitole

**Project Guide:** Prof. Aniket Shahade  
**Department:** Artificial Intelligence and Machine Learning  
**Institution:** Symbiosis Institute of Technology, Pune Campus, Symbiosis International (Deemed University), Pune, India.

---

*For questions about the implementation, open an issue in the repository or contact the project team through the academic channels approved by the institution.*
