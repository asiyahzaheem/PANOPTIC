# api/services/explanation_service.py
from __future__ import annotations
import numpy as np

SUBTYPE_NAMES = {
    0: "Squamous",
    1: "Pancreatic Progenitor",
    2: "ADEX",
    3: "Immunogenic",
}

def confidence_level(conf: float) -> str:
    if conf >= 0.80: return "high"
    if conf >= 0.60: return "medium"
    return "low"

def explain_conf(conf: float) -> str:
    lvl = confidence_level(conf)
    if lvl == "high":
        return "The model is fairly confident. Your inputs look similar to cases it learned from."
    if lvl == "medium":
        return "The model is moderately confident. Your inputs share patterns with more than one subtype."
    return "The model is not very confident. This can happen when the case does not strongly match one subtype."

def pct(x: float) -> str:
    return f"{x*100:.0f}%"

def modality_contrib(abs_attr: np.ndarray, z_dim: int) -> dict:
    img = float(abs_attr[:z_dim].sum())
    mol = float(abs_attr[z_dim:].sum())
    tot = img + mol + 1e-12
    return {"imaging": img / tot, "molecular": mol / tot}

def neighbors_summary(neighbors: list[dict], pred: int) -> str:
    if not neighbors:
        return "No similar past cases were available for comparison."
    same = sum(1 for n in neighbors if n.get("subtype_id") == pred)
    return f"Among the most similar past cases, {same}/{len(neighbors)} had the same subtype as this prediction."

def build_simple(pred: int, conf: float, probs_named: dict, contrib: dict, neighbors: list[dict]) -> dict:
    pred_name = SUBTYPE_NAMES.get(pred, str(pred))
    top3 = dict(sorted(probs_named.items(), key=lambda kv: -kv[1])[:3])
    return {
        "mode": "simple",
        "summary_text": (
            f"Predicted subtype: {pred_name}. "
            f"Model confidence: {conf:.2f} ({confidence_level(conf)}). "
            f"{explain_conf(conf)} "
            f"Main influence: CT {pct(contrib['imaging'])}, Molecular {pct(contrib['molecular'])}. "
            f"{neighbors_summary(neighbors[:5], pred)} "
            "This is a software prediction to support clinicians and may be wrong."
        ),
        "key_numbers": {
            "confidence": conf,
            "top3_probabilities": top3,
            "modality_contribution": {"ct": pct(contrib["imaging"]), "molecular": pct(contrib["molecular"])},
        },
        "nearest_neighbors": neighbors[:3],
    }

def build_detailed(pred: int, conf: float, probs_named: dict, contrib: dict, neighbors: list[dict], top_factors: list[dict]) -> dict:
    pred_name = SUBTYPE_NAMES.get(pred, str(pred))
    probs_sorted = sorted(probs_named.items(), key=lambda kv: -kv[1])
    margin = probs_sorted[0][1] - probs_sorted[1][1] if len(probs_sorted) > 1 else None
    return {
        "mode": "detailed",
        "summary_text": (
            f"Predicted subtype: {pred_name}. "
            f"Model confidence: {conf:.2f} ({confidence_level(conf)}). "
            f"{explain_conf(conf)} "
            f"CT contribution: {pct(contrib['imaging'])}; Molecular contribution: {pct(contrib['molecular'])}. "
            f"{neighbors_summary(neighbors[:8], pred)} "
            "This is not medical advice."
        ),
        "details": {
            "confidence": conf,
            "confidence_level": confidence_level(conf),
            "margin_top1_top2": float(margin) if margin is not None else None,
            "all_probabilities": probs_sorted,
            "nearest_neighbors": neighbors[:8],
            "top_internal_factors": top_factors[:25],
            "note": (
                "Top factors are internal embedding dimensions (z* for CT, emb_* for molecular). "
                "They improve transparency but are not direct clinical measurements."
            ),
        },
    }

