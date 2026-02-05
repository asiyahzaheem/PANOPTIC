# PanOptic

**Multimodal Pancreatic Cancer Subtype Classification** — A research prototype that predicts pancreatic ductal adenocarcinoma (PDAC) subtypes by combining CT imaging and molecular (gene expression) data through a graph neural network.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Pipeline Scripts (Run in Order)](#pipeline-scripts-run-in-order)
- [Running Locally](#running-locally)
- [Prediction](#prediction)
- [Running the Demo](#running-the-demo)
- [Testing Data](#testing-data)
- [Configuration](#configuration)
- [Data Pipeline & Scripts](#data-pipeline--scripts)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [PDAC Subtypes](#pdac-subtypes)
- [Disclaimer](#disclaimer)

---

## Overview

PanOptic classifies PDAC into four molecular subtypes:

- **Squamous**
- **Pancreatic Progenitor**
- **ADEX** (Aberrantly Differentiated Endocrine Exocrine)
- **Immunogenic**

The system takes two inputs per patient:

1. **CT scan** — NIfTI (`.nii`, `.nii.gz`) or DICOM (`.dcm`, `.dicom`) format
2. **Molecular data** — CSV or TSV with precomputed embeddings (`emb_0`–`emb_255`) or gene/value expression

It returns a predicted subtype, confidence score, and human-readable explanations (modality contribution, similar cases, alternatives).

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   CT Scan       │     │  Molecular Data │
│   (.nii/.dcm)   │     │  (.csv/.tsv)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ ResNet18        │     │ MolecularEmbedder│
│ (512-d vector)  │     │ (256-d vector)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Fusion Graph (GNN)    │
         │ GraphSAGE + k-NN      │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Subtype + Explanation │
         └───────────────────────┘
```

- **CT embedding**: K center slices → ResNet18 → mean-pooled 512-d vector
- **Molecular embedding**: Gene expression → MLP → 256-d vector (or direct `emb_0`–`emb_255` if precomputed)
- **Fusion**: Concatenated 768-d vector → z-score normalized → k-NN graph → GraphSAGE classifier
- **Explanations**: Gradient-based attribution (CT vs molecular contribution), similar-case comparison, alternative subtypes

---

## Project Structure

```
PANOPTIC/
├── pdac/                    # Core Python package
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # App entry point
│   │   ├── routers/         # Health, Predict endpoints
│   │   ├── services/        # CT embedder, mol parser, predictor, explanations
│   │   ├── schemas/         # Pydantic response models
│   │   └── core/            # Config, CORS
│   ├── src/                 # Data & models
│   │   ├── data/            # Dataset, preprocessing
│   │   ├── gnn/             # Graph construction (k-NN, standardization)
│   │   └── models/          # CNN backbone, GraphSAGE
│   └── molecular/           # Molecular embedder, subtype assignment
├── Frontend/                # React + Vite + shadcn/ui
│   └── src/components/      # Prototype, ResultsDisplay, AnalysisLoader
├── scripts/                 # Offline pipelines
│   ├── extractCTEmbeddings.py
│   ├── extract_molecular_embeddings.py
│   ├── fusionGraph.py
│   ├── trainGNN.py
│   └── predictCase.py       # Standalone CLI prediction
├── artifacts/               # Precomputed embeddings, fusion graph, labels
├── configs/
│   └── config.yaml          # Paths, preprocessing, GNN hyperparameters
└── models/                  # Trained GNN checkpoint (gnn_best.pt)
```

---

## Setup

### 1. Download Datasets

**CT and molecular datasets** must be downloaded from the **submission Google Drive** folder.

- **CT scans**: NIfTI format (`.nii` or `.nii.gz`), or DICOM if you will convert them
- **Molecular data**: Either a CCT file (genes × samples) for the full pipeline, or per-patient TSVs with `emb_0`–`emb_255` for prediction

Place the downloaded data in a directory of your choice (e.g. `PANOPTIC_DATA/`).

### 2. Configure Paths

Edit `configs/config.yaml` and set the data paths to match your layout:

| Config key | Description | Example |
|------------|-------------|---------|
| `data.project_root` | Absolute path to PANOPTIC repo | `"/Users/you/PANOPTIC"` |
| `data.base_dir` | Base directory for datasets | `"/path/to/PANOPTIC_DATA"` |
| `data.cptac_dicom_dir` | DICOM folders (if converting) | `"{base_dir}/CPTAC-PDA"` |
| `data.cptac_nifti_dir` | NIfTI CT scans | `"{base_dir}/cptac_nifti"` |
| `data.rnaseq_cct_dir` | RNA-seq CCT file | `"{base_dir}/cptac_rnaseq/RNAseq_Tumor.cct"` |
| `data.rnaseq_per_patient_tsv_dir` | Per-patient molecular TSVs | `"{base_dir}/cptac_rnaseq/per_patient_tsv"` |

Use absolute paths or paths relative to `project_root` as appropriate.

### 3. Backend (Python)

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch, torch-geometric, FastAPI, nibabel, pandas, scikit-learn, etc. (see `requirements.txt`)

### 4. Frontend (Node.js)

```bash
cd Frontend
npm install
# or: bun install
```

**Requirements**: Node.js 18+ (or Bun)

---

## Pipeline Scripts (Run in Order)

From a clean setup, run these scripts **in order** to build artifacts and train the model:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `convertCPTAC.py` | *(Optional)* Convert DICOM → NIfTI. Skip if CT data is already NIfTI. |
| 2 | `imagingIndex.py` | Build `imaging_index.csv` from NIfTI files in `cptac_nifti_dir` |
| 3 | `cctToTsv.py` | Split CCT into per-patient TSVs *(if using CCT)* |
| 4 | `prepareMolecular.py` | Build `molecular_index.csv` from CCT (patients × genes) |
| 5 | `exportMolecularCols.py` | Export `molecular_feature_cols.txt` from molecular index |
| 6 | `extractCTEmbeddings.py` | Extract 512-d CT embeddings → `imaging_embeddings.csv` |
| 7 | `extract_molecular_embeddings.py` | Extract 256-d molecular embeddings → `molecular_embeddings.csv`, `molecular_labels.csv` |
| 8 | `fusionGraph.py` | Merge imaging + molecular, build k-NN graph → `fusion_graph.pt` |
| 9 | `trainGNN.py` | Train GraphSAGE → `models/gnn_best.pt` |

**Example** (from project root):

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/imagingIndex.py
PYTHONPATH=. python scripts/prepareMolecular.py
PYTHONPATH=. python scripts/exportMolecularCols.py
PYTHONPATH=. python scripts/extractCTEmbeddings.py
PYTHONPATH=. python scripts/extract_molecular_embeddings.py
PYTHONPATH=. python scripts/fusionGraph.py
PYTHONPATH=. python scripts/trainGNN.py
```

**Note**: If your CT data is DICOM, run `convertCPTAC.py` first. If molecular data is per-patient TSVs, adapt `prepareMolecular.py` or create `molecular_index.csv` manually to match your layout.

---

## Running Locally

### Backend

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
PYTHONPATH=. uvicorn pdac.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd Frontend
npm run dev
```

### API URL

Create `Frontend/.env`:

```
VITE_API_BASE_URL=http://localhost:8000
```

If unset, the frontend defaults to `http://localhost:8000`.

### Open the App

Visit `http://localhost:5173` and scroll to the **Interactive Demo** section.

---

## Prediction

### Via API / Frontend (Raw CT + Molecular Files)

Upload a CT scan and molecular TSV via the web UI, or use curl:

```bash
curl -X POST \
  -F "ct_file=@patient1_scan.nii.gz" \
  -F "molecular_file=@patient1_mol.tsv" \
  -F "explain=detailed" \
  http://localhost:8000/predict
```

Use the **same patient’s** CT and molecular file for each request (see [Testing Data](#testing-data)).

### Via CLI (`predictCase.py`)

For pre-extracted embeddings or debugging with an existing patient:

```bash
# Using pre-extracted z (CT) and emb (molecular) CSV files
PYTHONPATH=. python scripts/predictCase.py \
  --z_csv path/to/z_embeddings.csv \
  --emb_csv path/to/molecular_embeddings.csv \
  --explain detailed

# Debug: run prediction using an existing patient from the fusion graph
PYTHONPATH=. python scripts/predictCase.py \
  --test_from_existing_patient C3N-00512 \
  --explain simple
```

---

## Running the Demo

### Local Hosting

See [Running Locally](#running-locally) for backend and frontend setup. Once both are running, visit `http://localhost:5173` and scroll to the **Interactive Demo** section. Use the 5 test pairs from the submission Google Drive (see [Testing Data](#testing-data)).

---

### PanOptic Online Domain (In Progress)

A hosted version is being deployed at **panopticai.online**.

- **URL**: https://panopticai.online (or https://www.panopticai.online)
- **Status**: In progress

When live, you can use the demo without running anything locally. The frontend will connect to the deployed API. Ensure `Frontend/.env` has:

```
VITE_API_BASE_URL=https://panoptic-render-1.onrender.com
```

(or whatever the production API URL is) when building for production.

---

## Testing Data

**5 pairs of test data** are provided in the **submission Google Drive** folder. Download these to run predictions.

Each pair consists of:

1. A **CT scan** (NIfTI format: `.nii` or `.nii.gz`)
2. A **molecular data file** (TSV format with `emb_0`–`emb_255` columns)

**Important**: Always use the **same patient’s** CT and TSV together. For example, `patient1_scan.nii.gz` must be paired with `patient1_mol.tsv` — never mix patients. The frontend validates that patient IDs match before analysis.

**Patient ID extraction rules** (from filenames):

- `patient_XXX` or `pt_XXX` → ID = `XXX`
- `XXX_something` (leading alphanumeric) → ID = `XXX`
- Whole filename (without extension) → ID = filename

---

## Configuration

Key settings in `configs/config.yaml`:

| Section | Key | Description |
|--------|-----|-------------|
| `data` | `project_root` | Project root path |
| | `artifacts_dir` | Output directory for embeddings, graph |
| | `fusion_graph_pt` | Fusion graph file |
| `preprocess` | `hu_min`, `hu_max` | HU window for CT (default -150 to 250) |
| | `k_slices` | Number of center slices per scan (default 8) |
| | `patch_crop` | Patch size for ResNet (default 224) |
| `gnn` | `knn_k` | k for k-NN graph (default 10) |
| | `num_classes` | PDAC subtypes (4) |
| | `temperature` | Softmax temperature for calibration (>1 = less overconfident) |

---

## Data Pipeline & Scripts

The offline pipeline (for retraining or rebuilding artifacts):

| Script | Purpose |
|--------|---------|
| `convertCPTAC.py` | DICOM → NIfTI conversion |
| `imagingIndex.py` | Build imaging index CSV |
| `extractCTEmbeddings.py` | CT → ResNet18 → 512-d embeddings |
| `prepareMolecular.py` / `cctToTsv.py` | Prepare molecular TSVs |
| `extract_molecular_embeddings.py` | Gene expression → 256-d embeddings |
| `fusionGraph.py` | Merge imaging + molecular, build k-NN graph, train/val/test splits |
| `trainGNN.py` | Train GraphSAGE on fusion graph |
| `predictCase.py` | CLI prediction (embeddings or existing patient ID) |

---

## API Reference

### Health

- `GET /health` — Returns `{"status": "ok"}`
- `GET /debug` — Returns paths and existence of fusion graph and model

### Predict

- `POST /predict`

**Request** (multipart/form-data):

- `ct_file` (file): CT scan — `.nii`, `.nii.gz`
- `molecular_file` (file): Molecular data — `.csv`, `.tsv` with `emb_0`–`emb_255` or gene/value
- `explain` (form): `"none"` | `"simple"` | `"detailed"`

**Response**:

```json
{
  "predicted_class": 0,
  "predicted_subtype": "Squamous",
  "confidence": 0.85,
  "confidence_level": "high",
  "probabilities": { "by_subtype_name": {...}, "by_class_id": {...} },
  "explanation": { ... },
  "notes": [ ... ]
}
```

---

## Frontend

- **Stack**: React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Framer Motion
- **Main components**: `Prototype` (upload + analysis), `ResultsDisplay` (subtype, confidence, explanations), `AnalysisLoader` (staged progress)
- **API base**: Set via `VITE_API_BASE_URL` in `.env`

---

## PDAC Subtypes

| ID | Subtype |
|----|---------|
| 0 | Squamous |
| 1 | Pancreatic Progenitor |
| 2 | ADEX |
| 3 | Immunogenic |

---

## Disclaimer

This is a **research prototype**. Results are for demonstration and should not be used for clinical decision-making. A qualified clinician should interpret any outputs in context.

---

## License

[Add your license here]
