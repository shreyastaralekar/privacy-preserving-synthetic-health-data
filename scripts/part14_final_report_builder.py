#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Python IDE conversion of Part14_DP_Medical_Final_Report_Presentation_Builder.ipynb.

Run from the project root with: python scripts/part14_final_report_builder.py
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
print("Part 14 – Part14_DP_Medical_Final_Report_Presentation_Builder")
print("Project root:", PROJECT_ROOT)
print("Storage mode:", STORAGE_MODE)
print("=" * 76 + "\n")

# # Part 14 – Final Research Report, Presentation and Deliverables Builder
#
# **Purpose:** generate the final editable research report (.docx) and presentation (.pptx) directly from the project’s saved settings, verified privacy records, evaluation metrics and plots.
#
# **Inputs:** outputs from Parts 1–13 in the shared Google Drive project folder.
#
# **Outputs:**
# - `final_deliverables/DP_Diffusion_Synthetic_Health_Data_Final_Report.docx`
# - `final_deliverables/DP_Diffusion_Synthetic_Health_Data_Presentation.pptx`
# - `final_deliverables/final_project_manifest.json`
# - An optional PDF export when LibreOffice is available.
# - A ZIP containing the report, presentation, manifest, trade-off data and report figures.
#
# > This generator never invents model performance or privacy results. If an experiment has not been completed, the report marks it as pending. Review institution-specific cover-page fields and verify all final interpretations before submission.

# ## 1. Install document-generation libraries and mount Google Drive

# [Notebook install command moved to requirements.txt]

from pathlib import Path
from datetime import datetime
import json, math, os, re, textwrap, shutil, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

PROJECT_ROOT = configure_project(__file__)
STORAGE_MODE = "Local IDE project folder"

DATA_DIR = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
METRICS_DIR = PROJECT_ROOT / 'metrics'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
SYNTHETIC_DIR = ARTIFACT_DIR / 'synthetic_data'
TRADEOFF_DIR = ARTIFACT_DIR / 'tradeoff_analysis'
FINAL_DIR = PROJECT_ROOT / 'final_deliverables'
for folder in [PROJECT_ROOT, DATA_DIR, PLOTS_DIR, CHECKPOINT_DIR, METRICS_DIR,
               ARTIFACT_DIR, SYNTHETIC_DIR, TRADEOFF_DIR, FINAL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = DATA_DIR / 'ddpm_config.json'
SCHEMA_PATH = DATA_DIR / 'schema_manifest.json'
if not CONFIG_PATH.exists() or not SCHEMA_PATH.exists():
    raise FileNotFoundError('Missing Part 5 configuration/schema. Complete Parts 1–5 before creating final deliverables.')
config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
TRADEOFF_CSV = TRADEOFF_DIR / 'part13_tradeoff_comparison.csv'
TRADEOFF_SUMMARY = TRADEOFF_DIR / 'part13_tradeoff_summary.json'
TRADEOFF_REPORT = TRADEOFF_DIR / 'part13_tradeoff_report.md'
tradeoff = pd.read_csv(TRADEOFF_CSV) if TRADEOFF_CSV.exists() else pd.DataFrame()
tradeoff_summary = json.loads(TRADEOFF_SUMMARY.read_text(encoding='utf-8')) if TRADEOFF_SUMMARY.exists() else {}

print('Storage:', STORAGE_MODE)
print('Project root:', PROJECT_ROOT)
print('Final output directory:', FINAL_DIR)
print('Part 13 results available:', not tradeoff.empty)

# ## 2. Collect available results and determine project completion status
#
# The report uses only files that are present. Missing artifacts are listed in the project status section, so a reader can distinguish an implementation plan from completed experiments.

def read_json(path):
    path = Path(path)
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        warnings.warn(f'Could not read {path.name}: {exc}')
        return None

schema_cols = list(schema.get('all_columns', []))
feature_cols = list(schema.get('feature_columns', []))
target_col = schema.get('target_column', 'DIABETE4')
config_summary = {
    'model_name': config.get('model_name', 'TabularDenoiser_Gaussian_DDPM'),
    'timesteps': config.get('timesteps'),
    'hidden_dim': config.get('hidden_dim'),
    'time_embed_dim': config.get('time_embed_dim'),
    'batch_size': config.get('batch_size'),
    'epochs': config.get('epochs'),
    'learning_rate': config.get('learning_rate'),
    'max_grad_norm': config.get('max_grad_norm'),
    'target_epsilons': config.get('target_epsilons', [1,3,8]),
    'accountant': config.get('accountant'),
    'secure_mode': config.get('secure_mode'),
    'train_rows': config.get('train_rows'),
    'test_rows': config.get('test_rows'),
    'input_dim': config.get('input_dim'),
    'data_fingerprint': config.get('data_fingerprint'),
}

status_files = {
    'preprocessing_metadata': PROJECT_ROOT / 'preprocessing_metadata.json',
    'EDA_summary': PROJECT_ROOT / 'eda_summary.json',
    'baseline_ML_summary': PROJECT_ROOT / 'baseline_ml_summary.json',
    'nonprivate_checkpoint': CHECKPOINT_DIR / 'ddpm_non_private.pt',
    'DP_accounting_verification': METRICS_DIR / 'privacy_accounting_verification.csv',
    'Part13_tradeoff_comparison': TRADEOFF_CSV,
}
status_files.update({
    'synthetic_datasets': SYNTHETIC_DIR,
    'Part10_fidelity_summaries': METRICS_DIR,
    'Part11_TSTR_TRTR_metrics': METRICS_DIR,
    'Part12_privacy_audits': METRICS_DIR,
})

status = {}
for key, path in status_files.items():
    if key == 'synthetic_datasets':
        status[key] = len(list(path.glob('synthetic_*.csv'))) > 0
    elif key == 'Part10_fidelity_summaries':
        status[key] = len(list(path.glob('part10_fidelity_summary_*.json'))) > 0
    elif key == 'Part11_TSTR_TRTR_metrics':
        status[key] = len(list(path.glob('part11_tstr_trtr_metrics_*.csv'))) > 0
    elif key == 'Part12_privacy_audits':
        status[key] = len(list(path.glob('part12_privacy_audit_summary_*.json'))) > 0
    else:
        status[key] = path.exists()

verification_path = METRICS_DIR / 'privacy_accounting_verification.csv'
verification = pd.read_csv(verification_path) if verification_path.exists() else pd.DataFrame()
verified_dp_count = 0
if not verification.empty and 'verification' in verification:
    verified_dp_count = int(((verification['verification'].astype(str) == 'PASS') &
                             (verification['target_budget_status'].astype(str) == 'WITHIN_TARGET')).sum())

model_labels = tradeoff['model_label'].astype(str).tolist() if 'model_label' in tradeoff.columns else []
synthetic_files = sorted(SYNTHETIC_DIR.glob('synthetic_*.csv'))
print('Available model labels:', model_labels or [p.stem.replace('synthetic_','') for p in synthetic_files])
print('Verified DP model/accountant rows:', verified_dp_count)

# ## 3. Prepare factual project narrative and results tables

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches as PptInches, Pt as PptPt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

RUN_DATE = datetime.now().strftime('%d %B %Y')
PROJECT_TITLE = 'Privacy-Preserving Synthetic Health Survey Data Generation Using Diffusion Models'
SUBTITLE = 'A Differentially Private Tabular DDPM Study Using the 2025 BRFSS Public-Use Dataset'

# Prepares a compact table for the Word report and slides.
if tradeoff.empty:
    results_table = pd.DataFrame([{'Status':'Pending','Details':'Run Parts 8 and 10–13 to populate measured comparison results.'}])
else:
    cols = [c for c in ['model_label','dp_accounting_status','achieved_epsilon','delta',
                        'mean_tvd_across_fields','mean_jsd_bits_across_fields',
                        'tstr_roc_auc','trtr_roc_auc','roc_auc_retention',
                        'exact_match_rate','distance_based_membership_attack_auc','pareto_efficient']
            if c in tradeoff.columns]
    results_table = tradeoff[cols].copy()

completed_parts = []
for part, check in [(1,PROJECT_ROOT/'preprocessing_metadata.json'), (2,PROJECT_ROOT/'preprocessing_metadata.json'),
                    (3,PROJECT_ROOT/'eda_summary.json'), (4,PROJECT_ROOT/'baseline_ml_summary.json'),
                    (5,DATA_DIR/'schema_manifest.json'), (6,CHECKPOINT_DIR/'ddpm_non_private.pt'),
                    (8,verification_path), (13,TRADEOFF_CSV)]:
    if check.exists(): completed_parts.append(part)

pending_status = [k for k,v in status.items() if not v]
if pending_status:
    execution_status = 'PARTIAL – one or more experiment artifacts are pending'
else:
    execution_status = 'ARTIFACTS AVAILABLE – review actual metrics and final checks before submission'

# Dynamic facts and caveats.
dataset_name = schema.get('dataset', 'CDC 2025 BRFSS')
privacy_unit = schema.get('privacy_unit', 'one respondent row')
all_field_count = len(schema_cols)
predictor_count = len(feature_cols)
train_rows = config.get('train_rows', 'Not recorded')
test_rows = config.get('test_rows', 'Not recorded')
input_dim = config.get('input_dim', schema.get('input_dim', 'Not recorded'))

# ## 4. Create original editable Word research report

def set_cell_shading(cell, fill='D9EAF7'):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def set_cell_text(cell, text, bold=False, color=None, size=8.5):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, dataframe, max_rows=30, font_size=8):
    if dataframe is None or dataframe.empty:
        doc.add_paragraph('No measured rows are available yet. Complete the corresponding experiment notebook.')
        return
    df = dataframe.head(max_rows).copy()
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    for j, col in enumerate(df.columns):
        set_cell_text(table.rows[0].cells[j], col.replace('_',' ').title(), bold=True, color=(255,255,255), size=font_size)
        set_cell_shading(table.rows[0].cells[j], '24476B')
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, value in enumerate(row.tolist()):
            if value is None or (np.isscalar(value) and pd.isna(value)): value = 'Not available'
            elif isinstance(value, (float, np.floating)): value = f'{value:.5f}'
            set_cell_text(cells[j], value, size=font_size)
    doc.add_paragraph('')
    return table


def add_bullet(doc, text):
    doc.add_paragraph(text, style='List Bullet')


def add_figure_if_exists(doc, path, caption, width=6.1):
    path = Path(path)
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in cp.runs:
            r.italic = True
            r.font.size = Pt(8)
    else:
        doc.add_paragraph(f'Figure pending: {caption} (run the relevant notebook to create this figure).')


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run('Page ')
    fldChar1 = OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText'); instrText.set(qn('xml:space'), 'preserve'); instrText.text = ' PAGE '
    fldChar2 = OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1); run._r.append(instrText); run._r.append(fldChar2)


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.72); sec.bottom_margin = Inches(0.72)
sec.left_margin = Inches(0.78); sec.right_margin = Inches(0.78)
styles = doc.styles
styles['Normal'].font.name = 'Aptos'
styles['Normal'].font.size = Pt(9.5)
styles['Normal'].paragraph_format.space_after = Pt(5)
for stylename, size, color in [('Title',24,(31,65,97)),('Heading 1',16,(31,65,97)),('Heading 2',12,(43,91,132)),('Heading 3',10,(43,91,132))]:
    st = styles[stylename]
    st.font.name = 'Aptos Display'
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(*color)

header = sec.header.paragraphs[0]
header.text = 'PRIVACY-PRESERVING SYNTHETIC HEALTH SURVEY DATA | RESEARCH REPORT'
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for r in header.runs: r.font.size = Pt(7); r.font.color.rgb = RGBColor(100,110,120)
add_page_number(sec.footer.paragraphs[0])

# Cover page
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('\n\n\nRESEARCH PROJECT REPORT\n').bold = True
p.runs[0].font.size = Pt(14); p.runs[0].font.color.rgb = RGBColor(43,91,132)
p = doc.add_paragraph(PROJECT_TITLE, style='Title'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph(SUBTITLE); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p.runs: r.font.size = Pt(12); r.italic = True
p = doc.add_paragraph('\n\nPrepared by: ____________________________________\nInstitution / Department: _________________________\nSupervisor: _____________________________________\nAcademic year: __________________________________')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph(f'Generated from saved project artifacts on {RUN_DATE}')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p.runs: r.font.size = Pt(8); r.font.color.rgb = RGBColor(100,100,100)
p = doc.add_paragraph(f'Execution status: {execution_status}')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p.runs: r.bold = True; r.font.color.rgb = RGBColor(140,80,35)
doc.add_page_break()

# Front matter
doc.add_heading('Document control and research integrity', level=1)
doc.add_paragraph('This is an editable working research report generated from project configuration and experiment artifacts. Replace the cover-page placeholders and inspect each automatically inserted result before formal submission. The report deliberately distinguishes completed measurements from planned procedures. It does not assert that an experiment ran when the corresponding output file is absent.')
doc.add_paragraph('Originality and attribution: the explanatory narrative is written for this project. Methods and external resources are identified in the references. A similarity score or AI-detector score cannot be guaranteed; institutional similarity and disclosure policies must be followed.')

doc.add_heading('Abstract', level=1)
abstract = (f'This project investigates privacy-preserving synthetic generation of categorical health-survey data using a compact Gaussian denoising diffusion probabilistic model (DDPM) trained with and without differentially private stochastic gradient descent (DP-SGD). The empirical source is the {dataset_name}, a public, self-reported telephone health survey. The encoded table uses {predictor_count} predictor fields and the target field {target_col}, giving {all_field_count} categorical variables and an encoded input dimension of {input_dim}. The workflow compares a non-private reference with DP training runs, verifies saved privacy-accountant states, generates synthetic records, measures distributional fidelity, evaluates one downstream prediction task using TSTR/TRTR, and conducts empirical privacy-risk diagnostics. The current artifact-derived status is: {execution_status.lower()}. Quantitative results are populated only when the upstream notebooks have saved the relevant metrics.')
doc.add_paragraph(abstract)

doc.add_heading('Keywords', level=2)
doc.add_paragraph('Synthetic data; tabular diffusion; differential privacy; DP-SGD; Opacus; BRFSS; privacy accounting; statistical fidelity; TSTR; membership inference.')

doc.add_heading('Contents', level=1)
for item in ['1. Introduction and problem statement','2. Aim, objectives and research questions','3. Dataset and data governance','4. Technical background','5. System architecture and methodology','6. Experimental design and evaluation metrics','7. Results and artifact status','8. Discussion and interpretation','9. Limitations, ethics and responsible use','10. Reproducibility and execution procedure','11. Conclusion and future work','References','Appendix A – Configuration and saved outputs']:
    add_bullet(doc, item)

doc.add_heading('1. Introduction and problem statement', level=1)
doc.add_paragraph('Health-survey microdata can support statistical analysis, software testing and machine-learning research, but record-level data can carry disclosure risks. Synthetic data offers a way to create artificial records that preserve selected statistical patterns. A synthetic dataset is not automatically private: a generator can memorise training examples, and empirical similarity alone cannot establish a privacy guarantee.')
doc.add_paragraph('This project studies a tabular diffusion model and integrates DP-SGD during model training. Differential privacy provides a mathematical bound on how much the training output can depend on any one privacy unit, under a stated adjacency definition and correctly implemented mechanism. The project therefore treats formal accountant verification and empirical disclosure audits as separate, complementary workstreams.')
doc.add_heading('Problem statement', level=2)
doc.add_paragraph('The central problem is to investigate whether a compact tabular diffusion workflow can produce useful synthetic BRFSS-like categorical records while providing a transparent, auditable respondent-level differential-privacy claim for the privately trained model. This is an experimental research question, not an assertion that the synthetic records are clinically valid or suitable for patient care.')

doc.add_heading('2. Aim, objectives and research questions', level=1)
doc.add_paragraph('Aim: design and evaluate a reproducible privacy-preserving synthetic-data pipeline based on a Gaussian tabular DDPM and Opacus DP-SGD.')
for x in [
    'Acquire and preprocess the 2025 BRFSS public-use dataset using a fixed categorical schema.',
    'Build a time-conditioned MLP denoiser and implement forward and reverse Gaussian diffusion over fixed one-hot feature blocks.',
    'Train a non-private reference model and separate DP-SGD models for configured target privacy budgets.',
    'Reconstruct saved Opacus accountant states and report achieved epsilon and delta rather than target labels alone.',
    'Generate synthetic tables and evaluate marginal and pairwise statistical fidelity.',
    'Compare downstream performance using TSTR and TRTR on the same held-out real test split.',
    'Audit exact matches, duplicate rows, distance-to-closest-record and a distance-based membership diagnostic.',
    'Analyse the utility–privacy trade-off and document reproducibility, limitations and safe-use boundaries.'
]: add_bullet(doc, x)
doc.add_heading('Research questions', level=2)
for x in [
    'How closely do the generated categorical marginals and pairwise associations reproduce the held-out BRFSS test distribution?',
    'How does downstream TSTR performance compare with TRTR under the same test split?',
    'What achieved (epsilon, delta) values are supported by saved and recomputed privacy-accountant states?',
    'How do fidelity and downstream utility vary across the verified DP models, and which configurations are non-dominated on the selected metrics?',
    'What do the empirical exact-match, distance and membership-risk audits reveal, and what can they not establish?'
]: add_bullet(doc, x)

doc.add_heading('3. Dataset and data governance', level=1)
doc.add_paragraph(f'The source configured in this project is {dataset_name}. The CDC describes the 2025 combined landline and cellular BRFSS public-use file as containing 356,158 records and 284 variables; the source is a self-reported telephone survey, not hospital electronic health records. The project narrows the analysis to {predictor_count} predictor fields and the {target_col} target, stored as {all_field_count} categorical fields in the Part 5 schema. The actual training and test row counts recorded in the project configuration are {train_rows} and {test_rows}.')
doc.add_paragraph('The CDC documents that the 2025 BRFSS aggregate file includes reporting areas that met the applicable minimum data requirements; the precise coverage and weighting details should be read from the annual overview and codebook before population-level interpretation. The project’s machine-learning split and compact feature subset are not a replacement for the survey’s complex sampling design or weighting procedures.')
doc.add_heading('Privacy unit and data boundary', level=2)
doc.add_paragraph(f'The configured privacy unit is {privacy_unit}. This means the DP analysis is intended to bound the influence of one respondent row under the adjacency convention implemented in the training pipeline. Because the source dataset is public-use microdata, the project demonstrates privacy-preserving training methodology; it does not retroactively make the public source confidential or remove all legal and ethical constraints on reuse.')
doc.add_heading('Target coding note', level=2)
doc.add_paragraph('For the downstream binary diabetes benchmark, the staged evaluation uses DIABETE4 code 1 as Yes and code 3 as No. Pregnancy-only diabetes code 2 and other special/refused/don’t-know codes are excluded rather than misclassified as No. The exact coding must be checked against the official year-specific codebook.')

doc.add_heading('4. Technical background', level=1)
doc.add_heading('4.1 Denoising diffusion probabilistic models', level=2)
doc.add_paragraph('A DDPM defines a forward process that gradually adds Gaussian noise to a data vector and a learned reverse process that removes noise. A time-conditioned neural network predicts the injected noise at a given diffusion timestep. In this project the input is a concatenation of fixed one-hot categorical blocks. This is a Gaussian DDPM over a one-hot representation, with argmax used to decode categorical blocks after sampling; it is not claimed to be a specialised categorical diffusion formulation.')
doc.add_heading('4.2 Differential privacy and DP-SGD', level=2)
doc.add_paragraph('DP-SGD clips per-example gradients to a configured norm and adds Gaussian noise to aggregated gradients before each optimizer update. Privacy loss depends on the sampling mechanism, sampling rate, clipping norm, noise multiplier, number of steps and accountant. The privacy budget is reported as (epsilon, delta). Smaller epsilon generally corresponds to a stronger bound for the same delta and adjacency definition, but the numerical value must be interpreted with the accountant and implementation assumptions.')
doc.add_heading('4.3 Opacus accounting', level=2)
doc.add_paragraph('The project saves accountant state and metadata from each DP run. Part 8 restores each accountant state using the same mechanism and recomputes epsilon at the recorded delta, then checks the result against the run metadata and target budget. The value verified from the saved accountant is the value to report. When multiple separately trained models using the same respondents are released, privacy losses must be composed; a single-model budget must not be represented as the total privacy cost of a multi-model release.')

doc.add_heading('5. System architecture and methodology', level=1)
doc.add_paragraph('The implementation is divided into staged Colab notebooks. Google Drive is mounted at the beginning of each notebook and the common project root is `MyDrive/DP_Medical_Synthetic_Project_2025/`. The persistent layout separates raw/processed data, checkpoints, model files, plots, metrics, generated synthetic tables and final deliverables.')
add_figure_if_exists(doc, PLOTS_DIR/'part5_denoiser_architecture.png', 'Figure 1. Time-conditioned tabular denoiser architecture, generated by Part 5.')
add_figure_if_exists(doc, PLOTS_DIR/'part5_ddpm_noise_schedule.png', 'Figure 2. Configured cosine noise schedule, generated by Part 5.')
add_figure_if_exists(doc, PLOTS_DIR/'part8_verified_privacy_budgets.png', 'Figure 3. Target versus recomputed privacy budget, generated by Part 8.')

doc.add_heading('5.1 Workflow stages', level=2)
stages = pd.DataFrame([
    ['1–2','Project setup, source-data preparation and preprocessing','Clean train/test tables and preprocessing metadata'],
    ['3–4','Exploratory analysis and baseline prediction','EDA summaries, plots and baseline metrics'],
    ['5','Fixed one-hot schema, DDPM architecture and diffusion schedule','Encoded arrays, schema manifest and model config'],
    ['6','Non-private reference training','Reference checkpoint and training history'],
    ['7–8','DP-SGD training and accountant verification','Private checkpoints, accountant states, verified epsilon/delta'],
    ['9','Reverse diffusion and categorical decoding','Synthetic CSV and generation metadata'],
    ['10','Marginal and pairwise statistical fidelity','TVD, JSD, association differences and figures'],
    ['11','TSTR/TRTR downstream task','Common-test-set classification metrics'],
    ['12','Empirical privacy-risk audits','Exact match, duplicates, DCR and membership diagnostic'],
    ['13','Integrated utility–privacy analysis','Consolidated comparison, plots and Pareto flags'],
    ['14','Research documentation and presentation','Editable DOCX, PPTX and manifest'],
], columns=['Parts','Process','Expected outputs'])
add_table(doc, stages, font_size=7.5)

doc.add_heading('5.2 Model and training configuration', level=2)
config_df = pd.DataFrame([
    ['Model', config_summary['model_name']], ['Diffusion timesteps', config_summary['timesteps']],
    ['Hidden dimension', config_summary['hidden_dim']], ['Time embedding dimension', config_summary['time_embed_dim']],
    ['Batch size', config_summary['batch_size']], ['Epochs', config_summary['epochs']],
    ['Learning rate', config_summary['learning_rate']], ['Gradient clipping norm', config_summary['max_grad_norm']],
    ['Target epsilon values', config_summary['target_epsilons']], ['Accountant', config_summary['accountant']],
    ['Secure RNG mode', config_summary['secure_mode']], ['Encoded dimension', config_summary['input_dim']],
], columns=['Setting','Configured value'])
add_table(doc, config_df, font_size=8)
doc.add_paragraph('These are configured experiment settings from the saved project configuration, not a claim that all training runs completed successfully. The run metadata and privacy-accountant outputs must be checked for each completed model.')

doc.add_heading('6. Experimental design and evaluation metrics', level=1)
doc.add_heading('6.1 Statistical fidelity', level=2)
doc.add_paragraph('For each categorical field, Part 10 compares category proportions between the synthetic sample and held-out real test set. Total variation distance is one half of the sum of absolute proportion differences. Jensen–Shannon divergence provides a symmetric distributional divergence. Pairwise Cramér’s V differences describe how well selected pairwise categorical associations are reproduced. These are empirical similarity measures and depend on the chosen schema, sample size and held-out split.')
doc.add_heading('6.2 Downstream utility: TSTR/TRTR', level=2)
doc.add_paragraph('Train-on-synthetic/test-on-real (TSTR) fits a classifier on the synthetic training table and evaluates it on the held-out real test records. Train-on-real/test-on-real (TRTR) provides a reference fit using a matched number of real training examples. The project records ROC-AUC, average precision, accuracy, balanced accuracy, precision, recall, F1 and specificity. The comparison is a benchmark on one selected binary task, not evidence of clinical validity or readiness for medical decision-making.')
doc.add_heading('6.3 Empirical privacy audits', level=2)
doc.add_paragraph('Part 12 checks exact synthetic-to-training matches, duplicate synthetic rows and nearest normalized Hamming distances. A simple distance-based membership diagnostic compares the distance-to-synthetic distributions of sampled training and held-out records. Results depend on the representation, distance metric and query budget. A low match rate or near-chance attack AUC does not prove that privacy risk is absent.')
doc.add_heading('6.4 Integrated trade-off analysis', level=2)
doc.add_paragraph('Part 13 preserves multiple dimensions rather than inventing a single weighted score. The Pareto frontier is computed among accountant-verified DP runs for which achieved epsilon, mean TVD and TSTR ROC-AUC are available. It marks models that are not dominated simultaneously on all three dimensions. It does not select a universally optimal model and must not obscure the trade-offs or release-composition requirements.')

doc.add_heading('7. Results and artifact status', level=1)
doc.add_paragraph(f'Current artifact-derived status: {execution_status}. The tables below are generated from the result files currently available in the Drive project folder. Empty or missing metrics are shown as unavailable, not replaced with estimates.')
doc.add_heading('7.1 Integrated model comparison', level=2)
add_table(doc, results_table, max_rows=20, font_size=6.8)
add_figure_if_exists(doc, PLOTS_DIR/'part13_epsilon_vs_tvd.png', 'Figure 4. Verified epsilon versus mean marginal TVD (Part 13).')
add_figure_if_exists(doc, PLOTS_DIR/'part13_epsilon_vs_tstr_auc.png', 'Figure 5. Verified epsilon versus downstream TSTR ROC-AUC (Part 13).')
add_figure_if_exists(doc, PLOTS_DIR/'part13_epsilon_vs_utility_retention.png', 'Figure 6. Verified epsilon versus TSTR/TRTR ROC-AUC retention (Part 13).')
add_figure_if_exists(doc, PLOTS_DIR/'part13_epsilon_vs_membership_auc.png', 'Figure 7. Verified epsilon versus empirical distance-membership AUC (Part 13).')
add_figure_if_exists(doc, PLOTS_DIR/'part13_artifact_completeness.png', 'Figure 8. Availability of evaluation artifacts for compared models (Part 13).')

# Include the key model-level plots generated in Parts 9, 11 and 12.
for label in model_labels:
    safe_label = str(label)
    add_figure_if_exists(doc, PLOTS_DIR/f'part9_synthetic_profile_{safe_label}.png',
                         f'Part 9 synthetic category profile for {safe_label}.', width=5.6)
    add_figure_if_exists(doc, PLOTS_DIR/f'part11_tstr_trtr_comparison_{safe_label}.png',
                         f'Part 11 TSTR/TRTR performance comparison for {safe_label}.', width=5.8)
    add_figure_if_exists(doc, PLOTS_DIR/f'part11_tstr_trtr_roc_{safe_label}.png',
                         f'Part 11 ROC curves on the real held-out test split for {safe_label}.', width=5.8)
    add_figure_if_exists(doc, PLOTS_DIR/f'part12_dcr_distribution_{safe_label}.png',
                         f'Part 12 distance-to-closest-training-record diagnostic for {safe_label}.', width=5.8)
    add_figure_if_exists(doc, PLOTS_DIR/f'part12_membership_distance_{safe_label}.png',
                         f'Part 12 distance-based membership diagnostic for {safe_label}.', width=5.8)

doc.add_heading('7.2 Privacy-accounting verification', level=2)
if verification.empty:
    doc.add_paragraph('No Part 8 privacy-accounting verification table was available at report-generation time. Complete Part 7 and Part 8 before making any formal DP epsilon/delta statement.')
else:
    cols = [c for c in ['model','target_epsilon','recorded_epsilon','recomputed_epsilon','delta','verification','target_budget_status','secure_mode'] if c in verification.columns]
    add_table(doc, verification[cols], max_rows=20, font_size=7)
    doc.add_paragraph('A verified row supports a reported accountant value only for the corresponding training mechanism, privacy unit, delta and data configuration. It does not automatically cover other separately trained models.')

doc.add_heading('7.3 Missing artifacts and required completion', level=2)
missing_df = pd.DataFrame([{'Artifact group':k.replace('_',' '),'Available':'Yes' if v else 'Pending'} for k,v in status.items()])
add_table(doc, missing_df, font_size=8)
if pending_status:
    doc.add_paragraph('Pending artifact groups: ' + '; '.join(pending_status) + '. Rerun the corresponding notebook and then rerun Parts 13 and 14 to refresh this report.')
else:
    doc.add_paragraph('The configured artifact groups were detected. Before submission, inspect every table, verify the selected release policy, and confirm that no metric is being misinterpreted.')

doc.add_heading('8. Discussion and interpretation', level=1)
if not tradeoff.empty:
    dp_rows = tradeoff[tradeoff.get('formal_dp_verified', pd.Series(False, index=tradeoff.index)).fillna(False)]
    doc.add_paragraph(f'The integrated table currently contains {len(tradeoff)} model label(s), of which {len(dp_rows)} are marked as accountant-verified by the Part 13 checks. This is an artifact status, not a guarantee about models whose metadata or accountant record is absent.')
    if not dp_rows.empty and dp_rows['mean_tvd_across_fields'].notna().any():
        best_tvd_row = dp_rows.loc[dp_rows['mean_tvd_across_fields'].idxmin()]
        doc.add_paragraph(f'Among the verified DP models with measured marginal fidelity, the lowest observed mean TVD is {best_tvd_row["mean_tvd_across_fields"]:.5f} for {best_tvd_row["model_label"]}. This describes only the evaluated split and feature schema.')
    if not dp_rows.empty and dp_rows['tstr_roc_auc'].notna().any():
        best_auc_row = dp_rows.loc[dp_rows['tstr_roc_auc'].idxmax()]
        doc.add_paragraph(f'Among verified DP models with a measured TSTR result, the largest observed ROC-AUC is {best_auc_row["tstr_roc_auc"]:.5f} for {best_auc_row["model_label"]}. Interpret it alongside the corresponding TRTR result, confidence/uncertainty considerations and class balance.')
    pareto_list = tradeoff.loc[tradeoff.get('pareto_efficient', pd.Series(False, index=tradeoff.index)).fillna(False), 'model_label'].astype(str).tolist() if 'pareto_efficient' in tradeoff.columns else []
    doc.add_paragraph('Pareto-efficient labels (for the selected three dimensions): ' + (', '.join(pareto_list) if pareto_list else 'none identified from the available complete measurements') + '.')
else:
    doc.add_paragraph('No integrated Part 13 comparison table was available. This section will be populated after the upstream evaluations and Part 13 have been run.')

doc.add_paragraph('Interpret findings as a balance among formal privacy, marginal and association fidelity, downstream utility and empirical risk diagnostics. A model with a lower epsilon may have lower fidelity or utility; a model with high TSTR utility may be less private or may reproduce weaker/stronger correlations than desired. The correct choice depends on the intended release setting and acceptable risk, not on any one metric in isolation.')

doc.add_heading('9. Limitations, ethics and responsible use', level=1)
for x in [
    'The BRFSS dataset is public-use, self-reported survey microdata. It is not an EHR dataset, and the project must not be described as training on hospital patient records.',
    'The selected 23 predictors plus DIABETE4 target are a reduced feature set. Results do not generalise automatically to the full 284-variable file, other years or other populations.',
    'The Gaussian DDPM operates over concatenated one-hot categorical vectors; argmax decoding is a pragmatic post-processing step and does not create a dedicated categorical diffusion model.',
    'The dataset is derived from a complex survey design. A simple train/test split and unweighted synthetic fidelity statistics do not reproduce all BRFSS design, weighting or variance-estimation requirements.',
    'Formal DP relies on the exact neighboring relation, privacy-unit definition, sampler, clipping/noise implementation, accountant, runtime code and correct handling of all releases. The configured secure_mode=False setting is not cryptographic RNG hardening.',
    'Releasing multiple models, generated datasets or intermediate artifacts may compose privacy loss. The release plan must be defined before distribution.',
    'TSTR/TRTR is one machine-learning benchmark and does not establish diagnostic accuracy, fairness, calibration or clinical suitability.',
    'Distance and membership audits are limited empirical attacks. Their failure to detect memorisation cannot prove privacy.',
    'Synthetic rows are artificial research artifacts, not real patients, and must not be used to make clinical decisions.'
]: add_bullet(doc, x)

doc.add_heading('10. Reproducibility and execution procedure', level=1)
for x in [
    'Open the staged notebooks in Google Colab and mount the same Google Drive account.',
    'Place the original CDC 2025 XPT file in the project data folder and complete Parts 1–4 in order.',
    'Run Part 5 once to create the fixed schema manifest and encoded train/test arrays. Do not change category order manually after model training.',
    'Run Part 6 for the non-private reference. Run Part 7 for each requested DP budget and retain the corresponding model, accountant state and run metadata.',
    'Run Part 8 to reconstruct accountant histories and verify achieved epsilon at the recorded delta.',
    'Run Part 9 for each model label intended for evaluation, then Parts 10, 11 and 12 with matching model labels.',
    'Run Part 13 to consolidate the measured trade-offs. Rerun it whenever upstream artifacts change.',
    'Run Part 14 again after all experiments are complete to refresh the Word report, PowerPoint, manifest and ZIP.'
]: add_bullet(doc, x)

doc.add_heading('11. Conclusion and future work', level=1)
doc.add_paragraph('This project establishes a staged and auditable workflow for studying synthetic categorical health-survey data generated by a tabular Gaussian DDPM, with a non-private reference and DP-SGD variants. Its strongest research contribution is the end-to-end integration of generation, accountant verification, statistical fidelity, downstream evaluation and empirical risk diagnostics in a persistent Colab/Google Drive workflow. The substantive conclusion must be written from the completed experimental tables rather than assumed in advance.')
doc.add_paragraph('Future work should consider a native categorical diffusion formulation, multiple random seeds and uncertainty intervals, survey-weight-aware analyses, richer downstream tasks, formal attack baselines, calibrated privacy accounting and secure randomness where the release requirements demand it. A broader evaluation should also test robustness across years, regions and feature subsets without treating the current public-use benchmark as confidential clinical data.')

doc.add_heading('References', level=1)
refs = [
    'Centers for Disease Control and Prevention. (2026). 2025 BRFSS Survey Data and Documentation. https://www.cdc.gov/brfss/annual_data/annual_2025.html',
    'Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33. https://arxiv.org/abs/2006.11239',
    'Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407. https://doi.org/10.1561/0400000042',
    'Opacus. (n.d.). PrivacyEngine API documentation. https://opacus.ai/api/privacy_engine.html',
    'Opacus. (n.d.). IAccountant API documentation. https://opacus.ai/api/accounting/iaccountant.html',
    'Centers for Disease Control and Prevention. (2026). BRFSS Annual Survey Data. https://www.cdc.gov/brfss/annual_data/annual_data.htm',
    'Scikit-learn developers. (n.d.). Model evaluation: scoring, metrics and performance. https://scikit-learn.org/stable/modules/model_evaluation.html'
]
for r in refs: doc.add_paragraph(r, style='List Number')

doc.add_heading('Appendix A – Configuration and saved outputs', level=1)
doc.add_paragraph('Shared persistent root: MyDrive/DP_Medical_Synthetic_Project_2025/')
for folder in ['data/ – source, processed tables, encoded arrays, schema and configuration',
               'checkpoints/ – non-private and DP model checkpoints, resume checkpoints and accountant states',
               'metrics/ – preprocessing, training, privacy accounting, fidelity, utility and audit CSV/JSON/Markdown files',
               'plots/ – EDA, training, fidelity, utility, privacy-audit and trade-off charts',
               'artifacts/synthetic_data/ – generated synthetic CSV files and metadata',
               'artifacts/tradeoff_analysis/ – consolidated utility–privacy comparison',
               'final_deliverables/ – this final report, presentation, manifest and package']:
    add_bullet(doc, folder)
doc.add_paragraph('Fingerprint of the configured training data: ' + str(config_summary['data_fingerprint']))
doc.add_paragraph('Report generation timestamp: ' + datetime.now().isoformat(timespec='seconds'))

REPORT_PATH = FINAL_DIR / 'DP_Diffusion_Synthetic_Health_Data_Final_Report.docx'
doc.save(REPORT_PATH)
print('Saved Word report:', REPORT_PATH)

# ## 5. Create an editable PowerPoint presentation

prs = Presentation()
prs.slide_width = PptInches(13.333)
prs.slide_height = PptInches(7.5)

NAVY = (31, 65, 97)
BLUE = (43, 91, 132)
GRAY = (90, 100, 110)
WHITE = (255, 255, 255)


def add_title(slide, title, subtitle=None):
    box = slide.shapes.add_textbox(PptInches(0.55), PptInches(0.25), PptInches(12.2), PptInches(0.75))
    tf = box.text_frame; tf.clear()
    p = tf.paragraphs[0]; p.text = title; p.font.size = PptPt(27); p.font.bold = True; p.font.color.rgb = __import__('pptx').dml.color.RGBColor(*NAVY)
    if subtitle:
        sub = slide.shapes.add_textbox(PptInches(0.6), PptInches(1.0), PptInches(12.0), PptInches(0.38))
        pp = sub.text_frame.paragraphs[0]; pp.text = subtitle; pp.font.size = PptPt(11); pp.font.color.rgb = __import__('pptx').dml.color.RGBColor(*GRAY)


def add_bullets(slide, items, top=1.55, font_size=19, left=0.9, width=11.6, height=5.3):
    box = slide.shapes.add_textbox(PptInches(left), PptInches(top), PptInches(width), PptInches(height))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for ix, item in enumerate(items):
        p = tf.paragraphs[0] if ix == 0 else tf.add_paragraph()
        p.text = str(item); p.level = 0; p.font.size = PptPt(font_size); p.space_after = PptPt(12)


def add_image_slide(title, image_path, caption=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, title, caption)
    p = Path(image_path)
    if p.exists():
        slide.shapes.add_picture(str(p), PptInches(1.0), PptInches(1.5), width=PptInches(11.3))
    else:
        add_bullets(slide, ['Figure not yet available. Run its source notebook and rerun Part 14.'])
    return slide

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, PROJECT_TITLE)
add_bullets(slide, [SUBTITLE, '', f'Dataset: {dataset_name}', f'Generated: {RUN_DATE}',
                    'Prepared by: ______________________________',
                    f'Execution status: {execution_status}'], top=2.0, font_size=21)

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Research problem and motivation')
add_bullets(slide, [
    'Synthetic data can support analytics and software testing, but generation alone does not guarantee privacy.',
    'A tabular diffusion model is trained with and without DP-SGD to compare utility and a formal privacy bound.',
    'Formal accountant verification and empirical disclosure-risk audits answer different questions.',
    'The goal is an auditable research workflow, not a clinical decision-support system.'
])

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Dataset and feature representation')
add_bullets(slide, [
    f'Source: {dataset_name} (public, self-reported telephone health survey; not hospital EHR).',
    f'Configured predictors: {predictor_count}; target: {target_col}; categorical fields total: {all_field_count}.',
    f'Encoded input dimension: {input_dim}; train rows: {train_rows}; held-out test rows: {test_rows}.',
    'One-hot category blocks are fixed by a schema manifest and reused throughout the pipeline.',
    'Survey weighting and complex-design inference are not fully reproduced by this compact experiment.'
])

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Model and privacy methodology')
add_bullets(slide, [
    f'Gaussian tabular DDPM with {config_summary["timesteps"]} diffusion timesteps and a time-conditioned MLP.',
    'Forward process adds Gaussian noise; reverse process predicts and removes noise.',
    'DP-SGD clips per-respondent gradients and adds calibrated Gaussian noise.',
    f'Accountant configured: {config_summary["accountant"]}; secure_mode: {config_summary["secure_mode"]}.',
    'Achieved epsilon is recomputed from the saved accountant state; target epsilon alone is not a result.'
])

add_image_slide('Tabular denoiser architecture', PLOTS_DIR/'part5_denoiser_architecture.png', 'Part 5 model schematic')
add_image_slide('Diffusion noise schedule', PLOTS_DIR/'part5_ddpm_noise_schedule.png', 'Configured cosine schedule; not a measured training result')

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Evaluation framework')
add_bullets(slide, [
    'Fidelity: marginal TVD/JSD and pairwise Cramér’s V differences.',
    'Utility: TSTR versus TRTR on the same held-out real test records.',
    'Empirical audit: exact matches, duplicate rows, nearest Hamming distance and distance-based membership AUC.',
    'Integrated analysis: achieved epsilon, fidelity, TSTR utility and Pareto-efficient configurations.',
    'No single metric proves privacy or clinical validity.'
])

add_image_slide('Verified privacy budgets', PLOTS_DIR/'part8_verified_privacy_budgets.png', 'Part 8: target budget compared with recomputed accountant epsilon')
add_image_slide('Utility–privacy: epsilon vs fidelity', PLOTS_DIR/'part13_epsilon_vs_tvd.png', 'Part 13: only accountant-verified DP runs with available measurements')
add_image_slide('Utility–privacy: epsilon vs TSTR ROC-AUC', PLOTS_DIR/'part13_epsilon_vs_tstr_auc.png', 'Part 13: common real held-out test split')
add_image_slide('Artifact completeness', PLOTS_DIR/'part13_artifact_completeness.png', 'Part 13: missing upstream results remain marked as pending')

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Current experimental status')
if tradeoff.empty:
    items = ['No Part 13 integrated result table was available at generation time.',
             'Run Parts 7–12 and then rerun Parts 13–14 to populate measured results.',
             'Do not state an achieved epsilon, synthetic utility score or risk-audit value until verified.']
else:
    items = [f'Model labels in comparison: {len(tradeoff)}',
             f'Accountant-verified DP models: {verified_dp_count}',
             f'Pareto-efficient labels: {", ".join(tradeoff.loc[tradeoff.get("pareto_efficient", pd.Series(False, index=tradeoff.index)).fillna(False), "model_label"].astype(str).tolist()) or "None identified"}',
             'See the accompanying Word report and CSV for per-model values.']
add_bullets(slide, items)

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Limitations and responsible use')
add_bullets(slide, [
    'Public BRFSS microdata are not confidential hospital records.',
    'The current model uses a compact categorical feature subset and Gaussian diffusion over one-hot vectors.',
    'Empirical audits are limited diagnostics; formal DP depends on the exact training mechanism and accountant.',
    'Multiple separately trained models released together require privacy composition.',
    'Synthetic outputs are research artifacts and must not be used for real clinical decisions.'
], font_size=17)

slide = prs.slides.add_slide(prs.slide_layouts[6]); add_title(slide, 'Conclusion and next steps')
add_bullets(slide, [
    'Complete all Colab experiments and confirm that each output is saved to Google Drive.',
    'Re-run Part 8 after DP training; report achieved epsilon and delta from verified accountant states.',
    'Re-run Parts 10–13 after any model regeneration or evaluation change.',
    'Inspect the final DOCX/PPTX, add author/institution details and check all claims before submission.',
    'Extend the study with repeated seeds, uncertainty intervals, survey-weight-aware analysis and categorical diffusion alternatives.'
])

PPT_PATH = FINAL_DIR / 'DP_Diffusion_Synthetic_Health_Data_Presentation.pptx'
prs.save(PPT_PATH)
print('Saved PowerPoint:', PPT_PATH)

# ## 6. Create a final manifest and package the deliverables

manifest = {
    'project_title': PROJECT_TITLE,
    'generated_at': datetime.now().isoformat(timespec='seconds'),
    'storage_mode': STORAGE_MODE,
    'project_root': str(PROJECT_ROOT),
    'dataset': dataset_name,
    'data_fingerprint': config.get('data_fingerprint'),
    'model_labels': model_labels,
    'verified_dp_model_rows': verified_dp_count,
    'execution_status': execution_status,
    'artifact_status': status,
    'tradeoff_summary': tradeoff_summary,
    'outputs': {
        'word_report': 'DP_Diffusion_Synthetic_Health_Data_Final_Report.docx',
        'presentation': 'DP_Diffusion_Synthetic_Health_Data_Presentation.pptx',
        'tradeoff_csv': str(TRADEOFF_CSV) if TRADEOFF_CSV.exists() else None,
        'tradeoff_report': str(TRADEOFF_REPORT) if TRADEOFF_REPORT.exists() else None,
    },
    'integrity_note': 'Only metrics in saved upstream artifact files are included; missing results are labeled pending.'
}
MANIFEST_PATH = FINAL_DIR / 'final_project_manifest.json'
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)), encoding='utf-8')

# Include final report, presentation, manifest, and selected evidence files/plots.
PACKAGE_PATH = FINAL_DIR / 'DP_Diffusion_Synthetic_Health_Data_Final_Deliverables.zip'
include_files = [
    REPORT_PATH, PPT_PATH, MANIFEST_PATH,
    TRADEOFF_CSV, TRADEOFF_SUMMARY, TRADEOFF_REPORT,
    PLOTS_DIR/'part5_denoiser_architecture.png', PLOTS_DIR/'part5_ddpm_noise_schedule.png',
    PLOTS_DIR/'part8_verified_privacy_budgets.png', PLOTS_DIR/'part13_epsilon_vs_tvd.png',
    PLOTS_DIR/'part13_epsilon_vs_tstr_auc.png', PLOTS_DIR/'part13_epsilon_vs_utility_retention.png',
    PLOTS_DIR/'part13_epsilon_vs_membership_auc.png', PLOTS_DIR/'part13_artifact_completeness.png'
]
with zipfile.ZipFile(PACKAGE_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in include_files:
        p = Path(p)
        if p.exists():
            # stable relative names inside package
            if p.parent == FINAL_DIR:
                arc = p.name
            elif p.parent == TRADEOFF_DIR:
                arc = 'tradeoff_analysis/' + p.name
            else:
                arc = 'plots/' + p.name
            zf.write(p, arcname=arc)

# Optional PDF export using LibreOffice if installed in the current runtime.
PDF_PATH = FINAL_DIR / 'DP_Diffusion_Synthetic_Health_Data_Final_Report.pdf'
lo = shutil.which('libreoffice') or shutil.which('soffice')
if lo:
    import subprocess
    try:
        subprocess.run([lo, '--headless', '--convert-to', 'pdf', '--outdir', str(FINAL_DIR), str(REPORT_PATH)],
                       check=True, capture_output=True, text=True, timeout=180)
        print('PDF export created:', PDF_PATH.exists())
    except Exception as exc:
        print('PDF export skipped:', exc)
else:
    print('PDF export not attempted: LibreOffice is not installed in this Colab runtime.')

print('\nFINAL OUTPUTS')
for p in [REPORT_PATH, PPT_PATH, MANIFEST_PATH, PACKAGE_PATH]:
    print(('OK' if p.exists() else 'MISSING'), p, f'({p.stat().st_size:,} bytes)' if p.exists() else '')

# ## 7. Final review checklist
#
# - [ ] Replace the author, institution, supervisor and academic-year placeholders in the Word report.
# - [ ] Review every dynamic result table against the original CSV/JSON files.
# - [ ] Confirm every DP epsilon and delta value corresponds to a Part 8 `PASS` and `WITHIN_TARGET` record.
# - [ ] Confirm whether one or multiple models/datasets will be released and apply privacy composition accordingly.
# - [ ] Read the CDC 2025 codebook and annual overview for exact variable coding and survey-design context.
# - [ ] Confirm that TSTR/TRTR conclusions are described as a limited downstream benchmark, not medical validation.
# - [ ] Rerun Parts 13 and 14 after any upstream metric changes.
# - [ ] Follow your institution’s citation, disclosure and originality requirements.
#
# **Important:** the Word report and PowerPoint are generated from current saved artifacts. If the experiments are not yet executed, they will show pending statuses and no invented performance values. This is the correct research practice.
