# PanOptic

**Multimodal Pancreatic Cancer Subtype Classification** — A research prototype that predicts pancreatic ductal adenocarcinoma (PDAC) subtypes by combining CT imaging and molecular (gene expression) data through a graph neural network.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
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

### Backend (Python)

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch, torch-geometric, FastAPI, nibabel, pandas, scikit-learn, etc. (see `requirements.txt`)

### Frontend (Node.js)

```bash
cd Frontend
npm install
# or: bun install
```

**Requirements**: Node.js 18+ (or Bun)

---

## Running the Demo

You can run the demo in two ways:

### Option 1: Local Hosting

Run both backend and frontend locally.

**1. Start the backend**

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
PYTHONPATH=. uvicorn pdac.api.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the frontend**

```bash
cd Frontend
npm run dev
```

**3. Configure API URL**

Create `Frontend/.env` (or use defaults):

```
VITE_API_BASE_URL=http://localhost:8000
```

If `VITE_API_BASE_URL` is not set, the frontend defaults to `http://localhost:8000`.

**4. Open the app**

Visit `http://localhost:5173` (or the port Vite shows) and scroll to the **Interactive Demo** section.

---

### Option 2: PanOptic Online Domain (In Progress)

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

**5 pairs of test data** are provided in the **submission Google Drive** folder.

Each pair consists of:

1. A **CT scan** (NIfTI format: `.nii` or `.nii.gz`)
2. A **molecular data file** (TSV format with `emb_0`–`emb_255` columns)

**Important**: Use the **same patient’s** CT and TSV together. The filenames must imply the same patient ID (e.g. `C3N-00512_scan.nii` and `C3N-00512_mol.tsv`). The frontend checks that patient IDs match before analysis.

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
