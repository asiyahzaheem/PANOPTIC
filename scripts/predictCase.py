"""
Predicts pancreatic cancer subtype for a new patient using the trained GNN model
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdac.src.models.gnnModel import GraphSAGEClassifier
from pdac.src.gnn.buildGraph import connect_to_train_edges

# map class IDs to readable subtype names
SUBTYPE_NAMES = {
    0: "Squamous",
    1: "Pancreatic Progenitor",
    2: "ADEX",
    3: "Immunogenic",
}

#Load the trained GNN model from checkpoint
def _load_model(model_pt: Path, in_dim: int, num_classes: int, hidden: int, dropout: float, device: torch.device):
    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model = GraphSAGEClassifier(in_dim=in_dim, hidden=hidden, num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# Model output logits to softmax probabilities
def _softmax_probs(logits: torch.Tensor) -> list[float]:
    return F.softmax(logits, dim=-1).detach().cpu().numpy().astype(float).tolist()

# Bucket confidence into high/medium/low for easier interpretation
def _confidence_level(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    if conf >= 0.60:
        return "medium"
    return "low"

def _pct(x: float) -> str:
    return f"{x*100:.0f}%"

#Plain lang explanation of what the confidence level means
def _explain_confidence(conf: float) -> str:
    level = _confidence_level(conf)
    if level == "high":
        return (
            "The model is fairly confident in this result. "
            "That usually means the input looks similar to cases it learned from."
        )
    if level == "medium":
        return (
            "The model is moderately confident. "
            "This means the input shares patterns with more than one subtype."
        )
    return (
        "The model is not very confident. "
        "This can happen when the input does not clearly match one subtype."
    )

# Explains whether CT or molecular data drove the decision
def _plain_language_why(modality_contrib: dict) -> str:

    img = modality_contrib.get("imaging", 0.0)
    mol = modality_contrib.get("molecular", 0.0)
    if img >= 0.65:
        return (
            "This decision was driven mostly by patterns found in the CT scan features, "
            "with molecular data playing a smaller role."
        )
    if mol >= 0.65:
        return (
            "This decision was driven mostly by patterns found in the molecular data, "
            "with CT scan features playing a smaller role."
        )
    return (
        "Both the CT scan features and molecular data contributed meaningfully to this decision."
    )

# Summarizes how similar past cases compare to the prediction
def _neighbors_plain_summary(neighbors: list[dict], pred: int) -> str:
    if not neighbors:
        return "No similar past cases were available to compare against."
    # how many match predicted subtype
    match = 0
    total = len(neighbors)
    for n in neighbors:
        if n.get("subtype_id", None) == pred:
            match += 1
    if "subtype_id" in neighbors[0]:
        return (
            f"Among the {total} most similar past cases, {match} had the same predicted subtype. "
            "This is one reason the model leans toward this result."
        )
    return (
        f"The model compared this case to {total} similar past cases. "
        "Similarity supports the prediction, but labels for neighbors were not available."
    )

# Build complete English summary for non-technical users
def _patient_friendly_simple_text(pred_name: str, conf: float, modality_contrib: dict, neighbors: list[dict], pred: int) -> str:
    return (
        f"**Predicted subtype:** {pred_name}\n"
        f"**Model confidence:** {conf:.2f} ({_confidence_level(conf)})\n\n"
        f"{_explain_confidence(conf)}\n\n"
        f"{_plain_language_why(modality_contrib)}\n\n"
        f"{_neighbors_plain_summary(neighbors, pred)}\n\n"
        "Important: this is a software prediction to support clinicians. It may be wrong and should not be used alone."
    )

# Format probabilities with both class IDs and subtype names
def _format_probabilities(probs: list[float]) -> dict:
    out = {}
    for i, p in enumerate(probs):
        out[str(i)] = float(p)
    out_named = {}
    for i, p in enumerate(probs):
        out_named[SUBTYPE_NAMES.get(i, str(i))] = float(p)
    return {"by_class_id": out, "by_subtype_name": out_named}

# Extract the list of patient IDs from the training set
def _train_patient_ids(fusion_pack: dict) -> list[str]:
    patient_ids_all = fusion_pack["patient_id"]
    idx_tr = fusion_pack["splits"]["train"]
    return [patient_ids_all[i] for i in idx_tr]

# Find the most similar training patients using cosine similarity
def _neighbors_explanation(
    x_tr_std: torch.Tensor,
    x_new_std: torch.Tensor,
    patient_ids_train: list[str],
    y_train: np.ndarray | None,
    topk: int = 5,
):
    a = x_tr_std.detach().cpu().numpy()
    b = x_new_std.detach().cpu().numpy()

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = (a_norm @ b_norm.T).reshape(-1)

    idx = np.argsort(-sims)[:topk]
    out = []
    for i in idx:
        item = {
            "patient_id": patient_ids_train[int(i)],
            "cosine_similarity": float(sims[int(i)]),
        }
        if y_train is not None:
            cls = int(y_train[int(i)])
            item["subtype_id"] = cls
            item["subtype_name"] = SUBTYPE_NAMES.get(cls, str(cls))
        out.append(item)

    return out, sims  # sims returned in case you want additional stats

# Compute gradient-based feature importance for the new patient
def _grad_for_new_node(model, x_trnew: torch.Tensor, edge_trnew: torch.Tensor, new_idx: int, target_class: int) -> np.ndarray:
    x = x_trnew.clone().detach().requires_grad_(True)
    logits = model(x, edge_trnew)[new_idx]
    score = logits[target_class]
    score.backward()
    g = x.grad[new_idx].detach().cpu().numpy().reshape(-1)
    return g

# Compute feature importance using integrated gradients
def _integrated_gradients_new_node(
    model,
    x_tr: torch.Tensor,
    edge_tr: torch.Tensor,
    x_new: torch.Tensor,
    edge_new_to_tr: torch.Tensor,
    steps: int,
    target_class: int,
):
    baseline = torch.zeros_like(x_new)
    attributions = torch.zeros_like(x_new)

    for s in range(1, steps + 1):
        alpha = s / steps
        x_interp_new = baseline + alpha * (x_new - baseline)

        x_trnew = torch.cat([x_tr, x_interp_new], dim=0).clone().detach().requires_grad_(True)
        edge_trnew = torch.cat([edge_tr, edge_new_to_tr], dim=1)

        new_idx = x_trnew.shape[0] - 1
        logits = model(x_trnew, edge_trnew)[new_idx]
        score = logits[target_class]
        score.backward()

        grad = x_trnew.grad[new_idx].detach()
        attributions += grad

    attributions = (x_new - baseline) * (attributions / steps)
    return attributions.detach().cpu().numpy().reshape(-1)

# Calculate how much CT vs molecular features contributed to the prediction
def _split_modality(abs_attr: np.ndarray, z_dim: int) -> dict:
    img = float(abs_attr[:z_dim].sum())
    mol = float(abs_attr[z_dim:].sum())
    total = img + mol + 1e-12
    return {"imaging": img / total, "molecular": mol / total}

# List the most important feature dimensions
def _top_features(abs_attr: np.ndarray, z_dim: int, topk: int):
    idx = np.argsort(-abs_attr)[:topk]
    top = []
    for j in idx:
        j = int(j)
        feat = f"z{j}" if j < z_dim else f"emb_{j - z_dim}"
        top.append({"feature": feat, "importance": float(abs_attr[j])})
    return top

# Build a simplified explanation for non-technical users
def _build_simple_explanation(
    *,
    probs: list[float],
    pred: int,
    x_tr_std: torch.Tensor,
    x_new_std: torch.Tensor,
    patient_ids_train: list[str],
    y_train: np.ndarray | None,
    abs_attr: np.ndarray,
    z_dim: int,
):
    conf = float(np.max(probs))

    neighbors, _ = _neighbors_explanation(
        x_tr_std, x_new_std, patient_ids_train=patient_ids_train, y_train=y_train, topk=5
    )

    contrib = _split_modality(abs_attr, z_dim=z_dim)

    # package key info for simple mode
    simple_numbers = {
        "confidence": conf,
        "confidence_level": _confidence_level(conf),
        "modality_contribution_percent": {
            "imaging": _pct(contrib["imaging"]),
            "molecular": _pct(contrib["molecular"]),
        },
        "top3_probabilities": dict(
            sorted(_format_probabilities(probs)["by_subtype_name"].items(), key=lambda kv: -kv[1])[:3]
        ),
    }

    pred_name = SUBTYPE_NAMES.get(pred, str(pred))
    patient_text = _patient_friendly_simple_text(
        pred_name=pred_name,
        conf=conf,
        modality_contrib=contrib,
        neighbors=neighbors,
        pred=pred,
    )

    return {
        "mode": "simple",
        "patient_friendly_summary": patient_text,
        "key_numbers": simple_numbers,
        "probabilities": _format_probabilities(probs),  # keep full distribution available
        "closest_similar_patients": neighbors[:3],      # short list for patient view
    }

# Build a detailed explanation with technical info for researchers
def _build_detailed_explanation(
    *,
    probs: list[float],
    pred: int,
    x_tr_std: torch.Tensor,
    edge_tr: torch.Tensor,
    x_new_std: torch.Tensor,
    edge_new_to_tr: torch.Tensor,
    patient_ids_train: list[str],
    y_train: np.ndarray | None,
    model,
    ig_steps: int,
    z_dim: int,
):
    conf = float(np.max(probs))

    at = _integrated_gradients_new_node(
        model=model,
        x_tr=x_tr_std,
        edge_tr=edge_tr,
        x_new=x_new_std,
        edge_new_to_tr=edge_new_to_tr,
        steps=ig_steps,
        target_class=pred,
    )
    abs_attr = np.abs(at)
    contrib = _split_modality(abs_attr, z_dim=z_dim)

    neighbors, _ = _neighbors_explanation(
        x_tr_std, x_new_std, patient_ids_train=patient_ids_train, y_train=y_train, topk=8
    )

    pred_name = SUBTYPE_NAMES.get(pred, str(pred))

    # start with plain-language summary
    patient_text = _patient_friendly_simple_text(
        pred_name=pred_name,
        conf=conf,
        modality_contrib=contrib,
        neighbors=neighbors[:5],
        pred=pred,
    )

    # add technical details for researchers
    probs_named = _format_probabilities(probs)["by_subtype_name"]
    probs_sorted = sorted(probs_named.items(), key=lambda kv: -kv[1])
    runner_up = probs_sorted[1][0] if len(probs_sorted) > 1 else None

    return {
        "mode": "detailed",
        "patient_friendly_summary": patient_text,
        "more_details": {
            "runner_up_subtype": runner_up,
            "modality_contribution_percent": {
                "imaging": _pct(contrib["imaging"]),
                "molecular": _pct(contrib["molecular"]),
            },
            "top_probabilities": probs_sorted,
            "closest_similar_patients": neighbors,
            # technical details for debugging and research
            "advanced": {
                "integrated_gradients_steps": int(ig_steps),
                "top_factors_internal_dimensions": _top_features(abs_attr, z_dim=z_dim, topk=25),
                "note": (
                    "Top factors are internal model dimensions (z* for CT, emb_* for molecular). "
                    "They are useful for debugging and research, but not directly interpretable as symptoms."
                ),
            },
        },
        "probabilities": _format_probabilities(probs),
        "confidence": conf,
        "confidence_level": _confidence_level(conf),
        "notes": [
            "Confidence values are model probabilities (softmax), not guaranteed correctness.",
            "This is not medical advice. A clinician should interpret results alongside other tests.",
        ],
    }


# Run prediction for a new patient given their CT and molecular embeddings
def predict_from_vectors(
    fusion_pack: dict,
    model_pt: Path,
    z_vec: np.ndarray,
    emb_vec: np.ndarray,
    explain: str = "simple",
    ig_steps: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load training graph (already standardized)
    gtr = fusion_pack["graphs"]["train"]
    x_tr_std = gtr["x"].to(device)
    edge_tr = gtr["edge_index"].to(device)

    # standardization params for RAW fused x
    mu = fusion_pack["standardize"]["mu"].to(device)
    sd = fusion_pack["standardize"]["sd"].to(device)

    # combine CT and molecular features into one vector
    z_vec = z_vec.astype(np.float32).reshape(1, -1)
    emb_vec = emb_vec.astype(np.float32).reshape(1, -1)
    z_dim = int(z_vec.shape[1])
    x_raw = np.concatenate([z_vec, emb_vec], axis=1).astype(np.float32)

    in_dim = int(x_tr_std.shape[1])
    if int(x_raw.shape[1]) != in_dim:
        raise ValueError(f"Feature dim mismatch: got {x_raw.shape[1]} expected {in_dim}")

    # z-score normalize using training set stats
    x_new_std = (torch.from_numpy(x_raw).to(device) - mu) / (sd + 1e-12)

    # connect new patient to nearest training patients in the graph
    k = 10
    metric = "cosine"
    edge_new_to_tr = connect_to_train_edges(
        x_tr_std.detach().cpu().numpy(),
        x_new_std.detach().cpu().numpy(),
        k=k,
        metric=metric
    ).to(device)

    # build combined graph with new patient added
    x_trnew = torch.cat([x_tr_std, x_new_std], dim=0)
    edge_trnew = torch.cat([edge_tr, edge_new_to_tr], dim=1)
    new_idx = x_trnew.shape[0] - 1

    # load model
    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    num_classes = int(cfg.get("gnn", {}).get("num_classes", 4))
    hidden = int(cfg.get("gnn", {}).get("hidden_dim", 128))
    dropout = float(cfg.get("gnn", {}).get("dropout", 0.3))

    model = _load_model(model_pt, in_dim=in_dim, num_classes=num_classes, hidden=hidden, dropout=dropout, device=device)

    # run the model to get prediction
    with torch.no_grad():
        logits = model(x_trnew, edge_trnew)[new_idx]
        probs = _softmax_probs(logits)
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

    result = {
        "predicted_class": pred,
        "predicted_subtype": SUBTYPE_NAMES.get(pred, str(pred)),
        "confidence": confidence,  # always present
        "confidence_level": _confidence_level(confidence),
        "probabilities": _format_probabilities(probs),  # always present
        "notes": [
            "Confidence values are model probabilities (softmax), not guaranteed correctness.",
        ],
    }

    # add explanation if requested
    if explain and explain != "none":
        patient_ids_train = _train_patient_ids(fusion_pack)

        # get training labels if available
        y_train = None
        try:
            y_all = fusion_pack["y"].detach().cpu().numpy()
            idx_tr = np.array(fusion_pack["splits"]["train"], dtype=int)
            y_train = y_all[idx_tr]
        except Exception:
            y_train = None

        if explain == "simple":
            # use fast gradient method for feature importance
            g = _grad_for_new_node(model, x_trnew, edge_trnew, new_idx, pred)
            abs_attr = np.abs(g)
            result["explanation"] = _build_simple_explanation(
                probs=probs,
                pred=pred,
                x_tr_std=x_tr_std,
                x_new_std=x_new_std,
                patient_ids_train=patient_ids_train,
                y_train=y_train,
                abs_attr=abs_attr,
                z_dim=z_dim,
            )

        elif explain == "detailed":
            result["explanation"] = _build_detailed_explanation(
                probs=probs,
                pred=pred,
                x_tr_std=x_tr_std,
                edge_tr=edge_tr,
                x_new_std=x_new_std,
                edge_new_to_tr=edge_new_to_tr,
                patient_ids_train=patient_ids_train,
                y_train=y_train,
                model=model,
                ig_steps=ig_steps,
                z_dim=z_dim,
            )
        else:
            raise ValueError("explain must be one of: none, simple, detailed")

    return result

# Load feature vector from JSON file
def _parse_vector_json(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "values" in obj:
        obj = obj["values"]
    return np.asarray(obj, dtype=np.float32)

# Extract feature vector from a single-row CSV file
def _parse_vector_csv(path: Path, prefix: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) == 0:
        raise ValueError(f"No columns starting with '{prefix}' found in {path}")

    if prefix == "z":
        cols = [c for c in cols if c[1:].isdigit()]
        cols = sorted(cols, key=lambda c: int(c[1:]))
    else:
        cols = [c for c in cols if c.startswith("emb_") and c.split("_")[1].isdigit()]
        cols = sorted(cols, key=lambda c: int(c.split("_")[1]))

    if len(df) != 1:
        raise ValueError(f"{path} must have exactly 1 row for single-case prediction.")
    return df.loc[0, cols].to_numpy(dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_graph", default="artifacts/fusion_graph.pt")
    ap.add_argument("--model", default="models/gnn_best.pt")
    ap.add_argument("--z_json", type=str, help="JSON list for imaging z0..zN (length matches training)")
    ap.add_argument("--emb_json", type=str, help="JSON list for molecular emb_0..emb_N (length matches training)")
    ap.add_argument("--z_csv", type=str, help="CSV with columns z0..zN single row")
    ap.add_argument("--emb_csv", type=str, help="CSV with columns emb_0..emb_N single row")
    ap.add_argument("--explain", default="simple", choices=["none", "simple", "detailed"])
    ap.add_argument("--ig_steps", default=32, type=int)
    ap.add_argument("--test_from_existing_patient", type=str,
                    help="Debug: use an existing patient_id from fusion_graph as input (no CT upload needed).")
    args = ap.parse_args()

    fusion_pack = torch.load(args.fusion_graph, map_location="cpu", weights_only=False)
    model_pt = Path(args.model)

    # debug mode: test with an existing patient from training data
    if args.test_from_existing_patient:
        pid = args.test_from_existing_patient.strip().upper()
        pid_list = [p.upper() for p in fusion_pack["patient_id"]]
        if pid not in pid_list:
            raise ValueError(f"patient_id {pid} not found in fusion_graph.pt")
        i = pid_list.index(pid)
        x_raw = fusion_pack["x"][i].numpy()

        # split into CT (first 512) and molecular (rest) features
        z_dim = 512
        z = x_raw[:z_dim]
        emb = x_raw[z_dim:]
    else:
        if args.z_json and args.emb_json:
            z = _parse_vector_json(Path(args.z_json))
            emb = _parse_vector_json(Path(args.emb_json))
        elif args.z_csv and args.emb_csv:
            z = _parse_vector_csv(Path(args.z_csv), prefix="z")
            emb = _parse_vector_csv(Path(args.emb_csv), prefix="emb_")
        else:
            raise ValueError("Provide either --z_json + --emb_json OR --z_csv + --emb_csv, or use --test_from_existing_patient")

    out = predict_from_vectors(
        fusion_pack=fusion_pack,
        model_pt=model_pt,
        z_vec=z,
        emb_vec=emb,
        explain=args.explain,
        ig_steps=args.ig_steps,
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
