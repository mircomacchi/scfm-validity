"""Inject known technical artefacts into a count matrix, so an embedding can be tested
on whether it separates them from real biology.

Two artefacts that routinely create "new cell types" in scRNA-seq:

* heterotypic doublets: two cells of different types captured in one droplet;
* ambient RNA: free transcripts from lysed cells added to an intact cell's profile.

Every synthetic cell is labelled in `obs["artefact"]` ("none", "doublet", "ambient"),
and doublets keep their parent types in `obs["parents"]`.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _downsample_rows(X: sp.csr_matrix, target: np.ndarray, rng: np.random.Generator):
    """Binomially thin each row of integer counts to about `target` total UMIs."""
    X = X.tocsr(copy=True)
    totals = np.asarray(X.sum(axis=1)).ravel()
    p = np.clip(target / np.maximum(totals, 1), 0, 1)
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        X.data[s:e] = rng.binomial(X.data[s:e].astype(np.int64), p[i])
    X.eliminate_zeros()
    return X


def simulate_doublets(
    adata: ad.AnnData,
    frac: float = 0.05,
    label_key: str = "cell_type",
    batch_key: str | None = None,
    depth_factor: float = 1.6,
    seed: int = 0,
) -> ad.AnnData:
    """Return `frac * n_obs` heterotypic doublets.

    Each doublet sums the counts of two cells with different `label_key` values
    (from the same batch when `batch_key` is given, as in a real droplet), then is
    thinned to `depth_factor` times the mean depth of its parents. Real doublets
    carry roughly 1.5 to 2 times the UMIs of a singlet, not the plain sum.
    """
    rng = np.random.default_rng(seed)
    n = int(round(frac * adata.n_obs))
    X = sp.csr_matrix(adata.X)
    labels = adata.obs[label_key].astype(str).to_numpy()
    batches = (
        adata.obs[batch_key].astype(str).to_numpy()
        if batch_key
        else np.zeros(adata.n_obs, dtype=int).astype(str)
    )

    a_idx, b_idx = [], []
    while len(a_idx) < n:
        a = rng.integers(adata.n_obs)
        pool = np.flatnonzero((batches == batches[a]) & (labels != labels[a]))
        if pool.size == 0:
            continue
        a_idx.append(a)
        b_idx.append(rng.choice(pool))
    a_idx, b_idx = np.array(a_idx), np.array(b_idx)

    summed = (X[a_idx] + X[b_idx]).tocsr()
    totals = np.asarray(X.sum(axis=1)).ravel()
    target = depth_factor * (totals[a_idx] + totals[b_idx]) / 2
    Xd = _downsample_rows(summed, target, rng)

    obs = adata.obs.iloc[a_idx].copy()
    obs.index = [f"doublet_{i}" for i in range(n)]
    obs["artefact"] = "doublet"
    obs["parents"] = [f"{labels[a]}+{labels[b]}" for a, b in zip(a_idx, b_idx, strict=True)]
    return ad.AnnData(X=Xd.astype(np.float32), obs=obs, var=adata.var.copy())


def ambient_profile(adata: ad.AnnData) -> np.ndarray:
    """Pooled expression profile of all cells, normalised to sum to 1.

    Real ambient RNA is estimated from empty droplets; without them, the pooled
    profile of the sample is the standard proxy.
    """
    p = np.asarray(sp.csr_matrix(adata.X).sum(axis=0)).ravel().astype(np.float64)
    return p / p.sum()


def simulate_ambient(
    adata: ad.AnnData,
    frac: float = 0.05,
    contamination: float = 0.3,
    batch_key: str | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Return `frac * n_obs` copies of real cells in which a share `contamination`
    of each cell's UMIs is replaced by draws from the ambient profile of its batch.
    """
    rng = np.random.default_rng(seed)
    n = int(round(frac * adata.n_obs))
    X = sp.csr_matrix(adata.X)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    totals = np.asarray(X[idx].sum(axis=1)).ravel()

    groups = (
        adata.obs[batch_key].astype(str).to_numpy()
        if batch_key
        else np.zeros(adata.n_obs, dtype=int).astype(str)
    )
    profiles = {g: ambient_profile(adata[groups == g]) for g in np.unique(groups[idx])}

    kept = _downsample_rows(X[idx], (1 - contamination) * totals, rng)
    rows = []
    for i, cell in enumerate(idx):
        n_amb = int(round(contamination * totals[i]))
        rows.append(rng.multinomial(n_amb, profiles[groups[cell]]))
    Xa = kept + sp.csr_matrix(np.vstack(rows), dtype=kept.dtype)

    obs = adata.obs.iloc[idx].copy()
    obs.index = [f"ambient_{i}" for i in range(n)]
    obs["artefact"] = "ambient"
    obs["parents"] = ""
    return ad.AnnData(X=Xa.astype(np.float32), obs=obs, var=adata.var.copy())


def inject_artefacts(
    adata: ad.AnnData,
    doublet_frac: float = 0.05,
    ambient_frac: float = 0.05,
    contamination: float = 0.3,
    label_key: str = "cell_type",
    batch_key: str | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Concatenate real cells with simulated doublets and ambient-contaminated cells."""
    real = adata.copy()
    real.obs["artefact"] = "none"
    real.obs["parents"] = ""
    parts = [real]
    if doublet_frac > 0:
        parts.append(simulate_doublets(adata, doublet_frac, label_key, batch_key, seed=seed))
    if ambient_frac > 0:
        parts.append(simulate_ambient(adata, ambient_frac, contamination, batch_key, seed=seed + 1))
    out = ad.concat(parts, join="outer", merge="same")
    out.var = adata.var.copy()
    out.obs["artefact"] = pd.Categorical(out.obs["artefact"], ["none", "doublet", "ambient"])
    return out
