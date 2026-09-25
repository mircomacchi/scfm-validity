"""Load a CELLxGENE-style h5ad, recover raw counts and subsample it to a laptop-sized AnnData."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _is_integer_counts(X, n: int = 2000) -> bool:
    """True if the first `n` stored values of X are non-negative integers."""
    data = X.data[:n] if sp.issparse(X) else np.asarray(X).ravel()[:n]
    return bool(np.all(data >= 0) and np.allclose(data, np.round(data)))


def stratified_indices(
    labels: pd.Series, n_cells: int, min_per_label: int = 100, seed: int = 0
) -> np.ndarray:
    """Sample about `n_cells` positions, keeping label proportions but at least
    `min_per_label` cells per label (or all of them, if fewer exist)."""
    rng = np.random.default_rng(seed)
    labels = labels.astype(str).to_numpy()
    uniq, counts = np.unique(labels, return_counts=True)
    frac = min(1.0, n_cells / len(labels))
    chosen = []
    for lab, cnt in zip(uniq, counts, strict=True):
        k = min(cnt, max(min_per_label, int(round(cnt * frac))))
        pos = np.flatnonzero(labels == lab)
        chosen.append(rng.choice(pos, size=k, replace=False))
    return np.sort(np.concatenate(chosen))


def load_subsample(
    path: str | Path,
    n_cells: int = 20_000,
    label_key: str = "cell_type",
    min_per_label: int = 100,
    seed: int = 0,
    obs_filter: dict[str, list[str]] | None = None,
) -> ad.AnnData:
    """Read `path` in backed mode, subsample by `label_key`, and return an in-memory
    AnnData with raw integer counts in `.X` and gene symbols in `var["feature_name"]`.

    CELLxGENE files keep normalised values in `.X` and raw counts in `.raw.X`;
    both layouts are handled.
    """
    backed = ad.read_h5ad(path, backed="r")
    obs = backed.obs
    mask = np.ones(backed.n_obs, dtype=bool)
    for key, values in (obs_filter or {}).items():
        mask &= obs[key].astype(str).isin(values).to_numpy()
    pos = np.flatnonzero(mask)
    idx = pos[stratified_indices(obs[label_key].iloc[pos], n_cells, min_per_label, seed)]

    sub = backed[idx].to_memory()
    backed.file.close()

    if sub.raw is not None and not _is_integer_counts(sub.X):
        raw = sub.raw.to_adata()
        raw.obs = sub.obs
        sub = raw
    if not _is_integer_counts(sub.X):
        raise ValueError("No raw integer counts found in .X or .raw.X")

    sub.X = sp.csr_matrix(sub.X, dtype=np.float32)
    if "feature_name" not in sub.var:
        sub.var["feature_name"] = sub.var_names
    sub.var["ensembl_id"] = sub.var_names.astype(str)
    sub.obs_names_make_unique()
    for col in sub.obs.select_dtypes("category"):
        sub.obs[col] = sub.obs[col].cat.remove_unused_categories()
    return sub
