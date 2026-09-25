"""Figures for the README: one UMAP row per embedding, plus a metric summary."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

ARTEFACT_COLOURS = {"none": "#d0d4da", "doublet": "#d1495b", "ambient": "#2e86ab"}


def umap_grid(
    adata: ad.AnnData, embedding_keys: list[str], label_key: str, out: str | Path, seed: int = 0
) -> Path:
    """UMAP of each embedding, coloured by cell type (left) and artefact (right)."""
    fig, axes = plt.subplots(len(embedding_keys), 2, figsize=(11, 4.6 * len(embedding_keys)))
    axes = np.atleast_2d(axes)
    order = np.argsort(adata.obs["artefact"].astype(str).ne("none").to_numpy(), kind="stable")
    for row, key in zip(axes, embedding_keys, strict=True):
        tmp = adata[order].copy()
        sc.pp.neighbors(tmp, use_rep=key, random_state=seed)
        sc.tl.umap(tmp, random_state=seed)
        sc.pl.umap(
            tmp,
            color=label_key,
            ax=row[0],
            show=False,
            frameon=False,
            legend_fontsize=6,
            title=f"{key}: {label_key}",
        )
        sc.pl.umap(
            tmp,
            color="artefact",
            ax=row[1],
            show=False,
            frameon=False,
            palette=ARTEFACT_COLOURS,
            title=f"{key}: artefact",
        )
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def artefact_bars(report: pd.DataFrame, out: str | Path) -> Path:
    """Bar chart of the three headline artefact metrics per embedding."""
    metrics = ["detectability_auroc", "phantom_fraction", "parent_consistency"]
    titles = [
        "Detectability (AUROC, higher = better)",
        "Artefacts in phantom clusters (lower = better)",
        "Doublets placed near a parent type (higher = better)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, metric, title in zip(axes, metrics, titles, strict=True):
        data = report.pivot(index="embedding", columns="artefact", values=metric)
        data = data.dropna(axis=1, how="all")
        data.plot.bar(ax=ax, color=[ARTEFACT_COLOURS[c] for c in data.columns], rot=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
