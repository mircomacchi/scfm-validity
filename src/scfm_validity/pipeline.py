"""In-memory pipeline shared by the CLI and the HTTP API."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from scfm_validity import artefacts, data, embed, metrics

# Methods the API may run in a request; Geneformer needs a GPU-sized budget and
# is left to the CLI and the batch scripts in hpc/.
FAST_METHODS = ("pca", "scvi")


def prepare_adata(
    source: str | Path,
    n_cells: int = 20_000,
    label_key: str = "cell_type",
    batch_key: str | None = "donor_id",
    min_genes: int = 200,
    doublet_frac: float = 0.05,
    ambient_frac: float = 0.05,
    contamination: float = 0.3,
    obs_filter: dict[str, list[str]] | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Subsample, filter low-quality cells and inject labelled artefacts."""
    adata = data.load_subsample(source, n_cells, label_key, seed=seed, obs_filter=obs_filter)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    full = artefacts.inject_artefacts(
        adata, doublet_frac, ambient_frac, contamination, label_key, batch_key, seed
    )
    full.uns["prepare"] = {
        "source": str(source),
        "label_key": label_key,
        "batch_key": batch_key,
        "seed": seed,
        "doublet_frac": doublet_frac,
        "ambient_frac": ambient_frac,
        "contamination": contamination,
    }
    return full


def run_report(
    source: str | Path,
    methods: tuple[str, ...] = ("pca",),
    n_cells: int = 20_000,
    label_key: str = "cell_type",
    batch_key: str | None = "donor_id",
    min_genes: int = 200,
    seed: int = 0,
    max_epochs: int = 100,
) -> pd.DataFrame:
    """Prepare, embed with `methods` and return the artefact metrics table."""
    unknown = set(methods) - set(FAST_METHODS)
    if unknown:
        raise ValueError(f"Unsupported methods {sorted(unknown)}; use {FAST_METHODS}")
    adata = prepare_adata(source, n_cells, label_key, batch_key, min_genes=min_genes, seed=seed)
    keys = []
    for m in methods:
        if m == "pca":
            keys.append(embed.embed_pca(adata, seed=seed))
        else:
            keys.append(embed.embed_scvi(adata, batch_key, max_epochs=max_epochs, seed=seed))
    return pd.concat([metrics.artefact_report(adata, k, label_key, seed) for k in keys])
