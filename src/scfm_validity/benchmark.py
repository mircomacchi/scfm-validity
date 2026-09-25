"""Standard integration benchmark (scib-metrics) on the real cells only."""

from __future__ import annotations

import anndata as ad
import pandas as pd
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation


def scib_report(
    adata: ad.AnnData,
    embedding_keys: list[str],
    batch_key: str,
    label_key: str = "cell_type",
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Bio-conservation and batch-correction scores for each embedding.

    Artefact cells are dropped first: scib metrics assume every label is a real
    cell type. PCA is the unintegrated reference for the PCR comparison.
    """
    real = adata[adata.obs["artefact"] == "none"].copy() if "artefact" in adata.obs else adata
    bm = Benchmarker(
        real,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=embedding_keys,
        pre_integrated_embedding_obsm_key="X_pca",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        n_jobs=n_jobs,
        progress_bar=False,
    )
    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    df = df.drop(index="Metric Type", errors="ignore").astype(float)
    df.index.name = "embedding"
    return df.reset_index()
