"""Artefact-validity metrics for a cell embedding.

The question: once doublets and ambient-contaminated cells are mixed in, does an
embedding keep them apart from real biology, or does it hand the analyst a
convincing "new cell type" that is only an artefact?

* `detectability_auroc`: can a linear classifier find the artefacts in the
  embedding? High is good: the embedding exposes them for QC.
* `knn_enrichment`: how much more often artefacts neighbour other artefacts than
  their prevalence predicts.
* `phantom_clusters`: after Leiden clustering, the share of artefact cells that
  sit in artefact-dominated clusters. These are the clusters an analyst would
  annotate as a novel population. High is bad.
* `doublet_parent_consistency`: share of doublets whose nearest real neighbours
  belong to one of their two parent types. High means the embedding places a
  doublet near the biology it came from.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def detectability_auroc(Z: np.ndarray, is_artefact: np.ndarray, seed: int = 0) -> float:
    """Five-fold cross-validated AUROC of a logistic regression on the embedding."""
    y = np.asarray(is_artefact, dtype=int)
    scores = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in folds.split(Z, y):
        scaler = StandardScaler().fit(Z[train])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(Z[train]), y[train])
        scores[test] = clf.predict_proba(scaler.transform(Z[test]))[:, 1]
    return float(roc_auc_score(y, scores))


def knn_enrichment(Z: np.ndarray, is_artefact: np.ndarray, k: int = 15) -> float:
    """Mean share of artefact neighbours around artefact cells, over artefact prevalence."""
    is_artefact = np.asarray(is_artefact, dtype=bool)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z[is_artefact])
    share = is_artefact[idx[:, 1:]].mean(axis=1).mean()
    return float(share / is_artefact.mean())


def phantom_clusters(
    Z: np.ndarray,
    is_artefact: np.ndarray,
    resolution: float = 1.0,
    threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Leiden-cluster the embedding and report artefact-dominated clusters."""
    is_artefact = np.asarray(is_artefact, dtype=bool)
    tmp = ad.AnnData(
        obs=pd.DataFrame(index=[str(i) for i in range(len(Z))]),
        obsm={"Z": np.asarray(Z, dtype=np.float32)},
    )
    sc.pp.neighbors(tmp, use_rep="Z", random_state=seed)
    sc.tl.leiden(tmp, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2)
    clusters = tmp.obs["leiden"].to_numpy()
    share = pd.Series(is_artefact).groupby(clusters).mean()
    phantom = share.index[share >= threshold]
    in_phantom = np.isin(clusters, phantom) & is_artefact
    return {
        "n_clusters": int(share.size),
        "n_phantom_clusters": int(phantom.size),
        "phantom_fraction": float(in_phantom.sum() / max(is_artefact.sum(), 1)),
    }


def doublet_parent_consistency(
    Z: np.ndarray, labels: np.ndarray, artefact: np.ndarray, parents: np.ndarray, k: int = 15
) -> float:
    """Share of doublets whose majority real-cell neighbour type is one of their parents."""
    real = artefact == "none"
    dbl = np.flatnonzero(artefact == "doublet")
    if dbl.size == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k).fit(Z[real])
    _, idx = nn.kneighbors(Z[dbl])
    real_labels = labels[real]
    hits = 0
    for row, d in zip(idx, dbl, strict=True):
        majority = pd.Series(real_labels[row]).mode().iloc[0]
        hits += majority in parents[d].split("+")
    return hits / dbl.size


def artefact_report(
    adata: ad.AnnData, embedding_key: str, label_key: str = "cell_type", seed: int = 0
) -> pd.DataFrame:
    """All artefact metrics for one embedding, one row per artefact type."""
    Z = np.asarray(adata.obsm[embedding_key])
    artefact = adata.obs["artefact"].astype(str).to_numpy()
    rows = []
    for kind in ("doublet", "ambient"):
        keep = np.isin(artefact, ["none", kind])
        if not (artefact == kind).any():
            continue
        is_art = artefact[keep] == kind
        row = {
            "embedding": embedding_key,
            "artefact": kind,
            "detectability_auroc": detectability_auroc(Z[keep], is_art, seed),
            "knn_enrichment": knn_enrichment(Z[keep], is_art),
            **phantom_clusters(Z[keep], is_art, seed=seed),
        }
        if kind == "doublet":
            row["parent_consistency"] = doublet_parent_consistency(
                Z[keep],
                adata.obs[label_key].astype(str).to_numpy()[keep],
                artefact[keep],
                adata.obs["parents"].astype(str).to_numpy()[keep],
            )
        rows.append(row)
    return pd.DataFrame(rows)
