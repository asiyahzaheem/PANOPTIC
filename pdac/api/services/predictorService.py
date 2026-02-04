# api/services/predictor_service.py
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn.functional as F

from pdac.src.models.gnnModel import GraphSAGEClassifier
from pdac.src.gnn.buildGraph import connect_to_train_edges

from pdac.api.services.explanationService import (
    SUBTYPE_NAMES, confidence_level, modality_contrib, build_simple, build_detailed
)

logger = logging.getLogger(__name__)

class PredictorService:
    def __init__(self, fusion_graph_pt: Path, model_pt: Path, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pack = torch.load(fusion_graph_pt, map_location="cpu", weights_only=False)

        gtr = self.pack["graphs"]["train"]
        self.x_tr_std = gtr["x"].to(self.device)
        self.edge_tr = gtr["edge_index"].to(self.device)

        self.mu = self.pack["standardize"]["mu"].to(self.device)
        self.sd = self.pack["standardize"]["sd"].to(self.device)

        # kNN settings from config (your yaml has these)
        self.k = int(cfg.get("gnn", {}).get("knn_k", 10))
        self.metric = str(cfg.get("gnn", {}).get("knn_metric", "cosine"))

        # train patient ids + labels for neighbor explanations
        patient_ids_all = self.pack["patient_id"]
        idx_tr = np.array(self.pack["splits"]["train"], dtype=int)
        self.patient_ids_train = [patient_ids_all[i] for i in idx_tr]
        self.y_train = self.pack["y"].detach().cpu().numpy()[idx_tr]

        # model hyperparams from config
        num_classes = int(cfg.get("gnn", {}).get("num_classes", 4))
        hidden = int(cfg.get("gnn", {}).get("hidden_dim", 128))
        dropout = float(cfg.get("gnn", {}).get("dropout", 0.5))

        in_dim = int(self.x_tr_std.shape[1])
        ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)

        self.model = GraphSAGEClassifier(in_dim=in_dim, hidden=hidden, num_classes=num_classes, dropout=dropout).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()
        self.temperature = float(cfg.get("gnn", {}).get("temperature", 1.0))

        logger.info(
            "PredictorService loaded: fusion_graph=%s model=%s temperature=%.2f",
            fusion_graph_pt, model_pt, self.temperature
        )

    def _neighbors(self, x_new_std: torch.Tensor, topk: int) -> list[dict]:
        a = self.x_tr_std.detach().cpu().numpy()
        b = x_new_std.detach().cpu().numpy()  # [1,D]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        sims = (a_norm @ b_norm.T).reshape(-1)
        idx = np.argsort(-sims)[:topk]
        out = []
        for i in idx:
            cls = int(self.y_train[int(i)])
            out.append({
                "patient_id": self.patient_ids_train[int(i)],
                "cosine_similarity": float(sims[int(i)]),
                "subtype_id": cls,
                "subtype_name": SUBTYPE_NAMES.get(cls, str(cls)),
            })
        return out

    def _abs_grad_attr(self, x_trnew: torch.Tensor, edge_trnew: torch.Tensor, new_idx: int, target_class: int) -> np.ndarray:
        x = x_trnew.clone().detach().requires_grad_(True)
        logits = self.model(x, edge_trnew)[new_idx]
        score = logits[target_class]
        score.backward()
        g = x.grad[new_idx].detach().cpu().numpy().reshape(-1)
        return np.abs(g)

    def _top_features(self, abs_attr: np.ndarray, z_dim: int, topk: int) -> list[dict]:
        idx = np.argsort(-abs_attr)[:topk]
        out = []
        for j in idx:
            j = int(j)
            feat = f"z{j}" if j < z_dim else f"emb_{j - z_dim}"
            out.append({"feature": feat, "importance": float(abs_attr[j])})
        return out

    def predict(self, z_vec: np.ndarray, emb_vec: np.ndarray, explain: str = "simple") -> dict:
        z_vec = z_vec.astype(np.float32).reshape(1, -1)
        emb_vec = emb_vec.astype(np.float32).reshape(1, -1)
        z_dim = int(z_vec.shape[1])

        x_raw = np.concatenate([z_vec, emb_vec], axis=1).astype(np.float32)
        in_dim = int(self.x_tr_std.shape[1])
        if int(x_raw.shape[1]) != in_dim:
            raise ValueError(f"Feature dim mismatch: got {x_raw.shape[1]} expected {in_dim}")

        x_new_std = (torch.from_numpy(x_raw).to(self.device) - self.mu) / (self.sd + 1e-12)

        edge_new_to_tr = connect_to_train_edges(
            self.x_tr_std.detach().cpu().numpy(),
            x_new_std.detach().cpu().numpy(),
            k=self.k,
            metric=self.metric
        ).to(self.device)

        x_trnew = torch.cat([self.x_tr_std, x_new_std], dim=0)
        edge_trnew = torch.cat([self.edge_tr, edge_new_to_tr], dim=1)
        new_idx = x_trnew.shape[0] - 1

        with torch.no_grad():
            logits = self.model(x_trnew, edge_trnew)[new_idx]
            # Temperature scaling: >1 reduces overconfidence
            probs = F.softmax(logits / self.temperature, dim=-1).detach().cpu().numpy().astype(float)
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))

        logger.info(
            "GNN prediction: pred=%s probs=%s confidence=%.4f",
            SUBTYPE_NAMES.get(pred, str(pred)), [f"{p:.3f}" for p in probs], conf
        )

        probs_named = {SUBTYPE_NAMES.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        neighbors = self._neighbors(x_new_std, topk=8 if explain == "detailed" else 5)

        explanation = None
        if explain != "none":
            abs_attr = self._abs_grad_attr(x_trnew, edge_trnew, new_idx, pred)
            contrib = modality_contrib(abs_attr, z_dim=z_dim)

            if explain == "simple":
                explanation = build_simple(pred, conf, probs_named, contrib, neighbors)
            elif explain == "detailed":
                top_factors = self._top_features(abs_attr, z_dim=z_dim, topk=25)
                explanation = build_detailed(pred, conf, probs_named, contrib, neighbors, top_factors)
            else:
                raise ValueError("explain must be one of: none, simple, detailed")

        return {
            "predicted_class": pred,
            "predicted_subtype": SUBTYPE_NAMES.get(pred, str(pred)),
            "confidence": conf,
            "confidence_level": confidence_level(conf),
            "probabilities": {
                "by_subtype_name": probs_named,
                "by_class_id": {str(i): float(p) for i, p in enumerate(probs)},
            },
            "explanation": explanation,
            "notes": [
                "Confidence values are model probabilities (softmax), not guaranteed correctness.",
                "This is not medical advice. A clinician should interpret results alongside other tests.",
            ],
        }

