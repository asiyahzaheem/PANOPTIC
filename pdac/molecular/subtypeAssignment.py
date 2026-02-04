"""
Assigns a subtype to each sample by averaging its signature genes and picking the highest scoring one
"""

import numpy as np
import pandas as pd
from pdac.molecular.subtypeSignatures import SUBTYPE_SIGNATURES

def assign_subtypes(df: pd.DataFrame):
    labels = []

    for _, row in df.iterrows():
        scores = {}
        for subtype, genes in SUBTYPE_SIGNATURES.items():
            present = [g for g in genes if g in df.columns]
            scores[subtype] = row[present].mean() if present else -np.inf # score subtpye by calculating the avg signature gene

        labels.append(max(scores, key=scores.get))

    return labels

