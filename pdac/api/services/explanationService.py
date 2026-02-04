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

def _confidence_narrative(conf: float) -> str:
    """Human-friendly explanation of what the confidence score means."""
    lvl = confidence_level(conf)
    pct_val = int(round(conf * 100))
    if lvl == "high":
        return (
            f"A {pct_val}% confidence score means the model is fairly confident in this result. "
            "Your scan and molecular data show patterns that closely match what the model learned from similar cases. "
            "While no prediction is certain, this level of confidence suggests the result is worth discussing with your care team."
        )
    if lvl == "medium":
        return (
            f"A {pct_val}% confidence score means the model is moderately confident. "
            "Your case shares features with more than one subtype, so the model is less certain. "
            "Your clinician may want to consider additional information or testing when interpreting this result."
        )
    return (
        f"A {pct_val}% confidence score means the model is not very confident. "
        "Your case does not clearly match one subtype based on the available data. "
        "This could mean your case is unusual, or that more information would help. "
        "Your care team should weigh this prediction alongside other clinical findings."
    )


def _modality_narrative(contrib: dict) -> str:
    """Human-friendly explanation of what drove the prediction."""
    img_pct = int(round(contrib["imaging"] * 100))
    mol_pct = int(round(contrib["molecular"] * 100))
    if img_pct >= 65:
        return (
            "The prediction was driven mainly by patterns in your CT scan. "
            f"Imaging features contributed {img_pct}% of the model's decision, while molecular (genetic) data contributed {mol_pct}%. "
            "This suggests the visual patterns in your scan were the strongest signal for this subtype."
        )
    if mol_pct >= 65:
        return (
            "The prediction was driven mainly by your molecular (genetic) data. "
            f"Molecular features contributed {mol_pct}% of the model's decision, while CT scan patterns contributed {img_pct}%. "
            "This suggests the gene expression profile was the strongest signal for this subtype."
        )
    return (
        "Both your CT scan and molecular data contributed meaningfully to this prediction. "
        f"Imaging contributed {img_pct}% and molecular data contributed {mol_pct}%. "
        "The model combined information from both sources to reach its conclusion."

    )


def _neighbors_narrative(neighbors: list[dict], pred: int) -> tuple[str, list[dict]]:
    """Human-friendly explanation of similar cases, plus a simplified list for display."""
    if not neighbors:
        return "No similar past cases were available for comparison.", []
    same = sum(1 for n in neighbors if n.get("subtype_id") == pred)
    total = len(neighbors)
    match_pct = int(round(100 * same / total)) if total else 0
    narrative = (
        f"The model compared your case to {total} similar cases from its database. "
        f"{same} of those {total} cases ({match_pct}%) had the same subtype as this prediction. "
    )
    if same >= total * 0.75:
        narrative += "This strong agreement among similar cases supports the result."
    elif same >= total * 0.5:
        narrative += "This moderate agreement provides some support for the result."
    else:
        narrative += "The mixed agreement suggests some uncertainty; your clinician may want to consider this when interpreting the result."
    # Build a simple list for display (no raw IDs or technical fields)
    display_list = []
    for n in neighbors[:8]:
        subtype = n.get("subtype_name", "Unknown")
        sim = n.get("cosine_similarity", 0)
        if sim >= 0.9:
            sim_label = "Very similar"
        elif sim >= 0.7:
            sim_label = "Similar"
        else:
            sim_label = "Somewhat similar"
        matches = "✓ Matches prediction" if n.get("subtype_id") == pred else "Different subtype"
        display_list.append({
            "subtype": subtype,
            "similarity": sim_label,
            "matches_prediction": n.get("subtype_id") == pred,
        })
    return narrative, display_list


def _alternatives_narrative(probs_sorted: list[tuple], pred_name: str) -> tuple[str, list[dict]]:
    """Human-friendly explanation of other subtypes considered."""
    others = [(n, p) for n, p in probs_sorted if n != pred_name and p > 0.01]
    if not others:
        return "The model did not seriously consider other subtypes for this case.", []
    narrative = (
        f"The model also considered other subtypes. "
        f"The second most likely was {others[0][0]} ({int(round(others[0][1] * 100))}%). "
    )
    if len(others) > 1:
        rest = ", ".join(f"{n} ({int(round(p * 100))}%)" for n, p in others[1:3])
        narrative += f" Others considered: {rest}. "
    narrative += "Your clinician can discuss what these alternatives might mean for your care."
    display_list = [{"name": n, "percentage": int(round(p * 100))} for n, p in others[:4]]
    return narrative, display_list


def build_detailed(pred: int, conf: float, probs_named: dict, contrib: dict, neighbors: list[dict], top_factors: list[dict]) -> dict:
    pred_name = SUBTYPE_NAMES.get(pred, str(pred))
    probs_sorted = sorted(probs_named.items(), key=lambda kv: -kv[1])
    margin = probs_sorted[0][1] - probs_sorted[1][1] if len(probs_sorted) > 1 else None

    # Build human-friendly narrative sections
    neighbors_narrative, similar_cases = _neighbors_narrative(neighbors[:8], pred)
    alternatives_narrative, alternatives_list = _alternatives_narrative(probs_sorted, pred_name)

    sections = [
        {
            "heading": "Understanding your confidence score",
            "body": _confidence_narrative(conf),
            "highlight": f"{int(round(conf * 100))}% — {confidence_level(conf).title()} confidence",
        },
        {
            "heading": "What drove this prediction?",
            "body": _modality_narrative(contrib),
            "breakdown": f"CT scan: {pct(contrib['imaging'])}  •  Molecular data: {pct(contrib['molecular'])}",
        },
        {
            "heading": "Comparison to similar cases",
            "body": neighbors_narrative,
            "similar_cases": similar_cases,
        },
        {
            "heading": "Other subtypes considered",
            "body": alternatives_narrative,
            "alternatives": alternatives_list,
        },
    ]

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
            "sections": sections,
            # Keep raw data for debugging / future use
            "confidence": conf,
            "confidence_level": confidence_level(conf),
            "margin_top1_top2": float(margin) if margin is not None else None,
            "all_probabilities": probs_sorted,
            "nearest_neighbors": neighbors[:8],
            "top_internal_factors": top_factors[:25],
        },
    }

