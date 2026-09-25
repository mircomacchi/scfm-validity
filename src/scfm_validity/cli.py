"""Command-line entry point: prepare -> embed -> evaluate."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import typer

from scfm_validity import artefacts, data, embed, metrics, plots

app = typer.Typer(add_completion=False, help=__doc__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scfm_validity")

EMBEDDERS = {"pca": "X_pca", "scvi": "X_scvi", "geneformer": "X_geneformer"}


@app.command()
def prepare(
    source: Path = typer.Argument(..., help="Input h5ad (e.g. a CELLxGENE download)."),
    out: Path = typer.Argument(..., help="Output h5ad with real and artefact cells."),
    n_cells: int = 20_000,
    label_key: str = "cell_type",
    batch_key: str = "donor_id",
    min_genes: int = 200,
    doublet_frac: float = 0.05,
    ambient_frac: float = 0.05,
    contamination: float = 0.3,
    obs_filter: str = typer.Option("{}", help='JSON, e.g. \'{"suspension_type": ["cell"]}\''),
    seed: int = 0,
) -> None:
    """Subsample, filter low-quality cells and inject labelled artefacts."""
    adata = data.load_subsample(
        source, n_cells, label_key, seed=seed, obs_filter=json.loads(obs_filter)
    )
    sc.pp.filter_cells(adata, min_genes=min_genes)
    log.info("Real cells after QC: %d", adata.n_obs)
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
    out.parent.mkdir(parents=True, exist_ok=True)
    full.write_h5ad(out, compression="gzip")
    log.info("Wrote %s: %s", out, full.obs["artefact"].value_counts().to_dict())


@app.command("embed")
def embed_cmd(
    h5ad: Path,
    methods: str = typer.Option("pca,scvi,geneformer", help="Comma-separated."),
    max_epochs: int = 100,
    geneformer_model: str = "gf-12L-38M-i4096",
    seed: int = 0,
) -> None:
    """Add embeddings to the h5ad in place."""
    adata = ad.read_h5ad(h5ad)
    batch_key = adata.uns["prepare"]["batch_key"]
    for m in methods.split(","):
        log.info("Embedding with %s", m)
        if m == "pca":
            embed.embed_pca(adata, seed=seed)
        elif m == "scvi":
            embed.embed_scvi(adata, batch_key, max_epochs=max_epochs, seed=seed)
        elif m == "geneformer":
            embed.embed_geneformer(adata, model_name=geneformer_model)
        else:
            raise typer.BadParameter(f"Unknown method {m!r}; choose from {list(EMBEDDERS)}")
    adata.write_h5ad(h5ad, compression="gzip")


@app.command()
def evaluate(
    h5ad: Path,
    outdir: Path,
    scib: bool = typer.Option(True, help="Also run the scib-metrics benchmark."),
    seed: int = 0,
) -> None:
    """Score every embedding in the file; write parquet tables and figures."""
    adata = ad.read_h5ad(h5ad)
    label_key = adata.uns["prepare"]["label_key"]
    batch_key = adata.uns["prepare"]["batch_key"]
    keys = [k for k in EMBEDDERS.values() if k in adata.obsm]
    outdir.mkdir(parents=True, exist_ok=True)

    report = pd.concat([metrics.artefact_report(adata, k, label_key, seed) for k in keys])
    report.to_parquet(outdir / "artefact_metrics.parquet", index=False)
    log.info("\n%s", report.to_string(index=False))

    if scib:
        from scfm_validity.benchmark import scib_report

        scores = scib_report(adata, keys, batch_key, label_key)
        scores.to_parquet(outdir / "scib_metrics.parquet", index=False)
        log.info("\n%s", scores.to_string(index=False))

    plots.artefact_bars(report, outdir / "artefact_metrics.png")
    plots.umap_grid(adata, keys, label_key, outdir / "umap_grid.png", seed)


if __name__ == "__main__":
    app()
