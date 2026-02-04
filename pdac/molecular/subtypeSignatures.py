"""
Gene signatures for each pancreatic cancer subtype. Each subtype is defined by a list of marker genes
"""

SUBTYPE_SIGNATURES = {
    "Squamous": [
        "KRT5", "KRT6A", "TP63", "S100A2", "LAMC2"
    ],
    "Pancreatic_Progenitor": [
        "PDX1", "HNF1A", "FOXA2", "GATA6"
    ],
    "Immunogenic": [
        "CXCL9", "CXCL10", "CD3D", "CD8A", "LCK"
    ],
    "ADEX": [
        "NR5A2", "RBPJL", "CEL", "PRSS1"
    ],
}
