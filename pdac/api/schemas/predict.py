# api/schemas/predict.py
from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class PredictResponse(BaseModel):
    predicted_class: int
    predicted_subtype: str
    confidence: float
    confidence_level: str
    probabilities: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None
    notes: List[str] = []

