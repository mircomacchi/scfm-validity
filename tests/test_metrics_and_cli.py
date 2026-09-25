import anndata as ad
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from scfm_validity import artefacts, data, embed, metrics
from scfm_validity.cli import app


def test_separable_artefacts_score_high():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 1, (300, 5)), rng.normal(6, 1, (30, 5))])
    is_art = np.r_[np.zeros(300, bool), np.ones(30, bool)]
    assert metrics.detectability_auroc(Z, is_art) > 0.95
    assert metrics.knn_enrichment(Z, is_art) > 5
    assert metrics.phantom_clusters(Z, is_art)["phantom_fraction"] > 0.9


def test_mixed_artefacts_score_low():
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 1, (330, 5))
    is_art = np.r_[np.zeros(300, bool), np.ones(30, bool)]
    assert metrics.detectability_auroc(Z, is_art) < 0.7
    assert metrics.phantom_clusters(Z, is_art)["phantom_fraction"] < 0.3


def test_stratified_indices_keep_rare_labels():
    labels = pd.Series(["common"] * 1000 + ["rare"] * 20)
    idx = data.stratified_indices(labels, n_cells=100, min_per_label=15)
    picked = labels.iloc[idx].value_counts()
    assert picked["rare"] == 15 and picked["common"] == 98


def test_artefact_report_on_pca(toy_adata):
    full = artefacts.inject_artefacts(toy_adata, 0.1, 0.1, batch_key="donor_id")
    key = embed.embed_pca(full, n_hvg=200, n_comps=10)
    report = metrics.artefact_report(full, key)
    assert set(report["artefact"]) == {"doublet", "ambient"}
    assert report["detectability_auroc"].between(0, 1).all()


def test_cli_end_to_end_pca(toy_adata, tmp_path):
    src = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(src)
    prepared = tmp_path / "prepared.h5ad"
    runner = CliRunner()
    r = runner.invoke(
        app, ["prepare", str(src), str(prepared), "--n-cells", "600", "--min-genes", "10"]
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["embed", str(prepared), "--methods", "pca"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["evaluate", str(prepared), str(tmp_path / "out"), "--no-scib"])
    assert r.exit_code == 0, r.output
    table = pd.read_parquet(tmp_path / "out" / "artefact_metrics.parquet")
    assert list(table["embedding"].unique()) == ["X_pca"]
    assert (tmp_path / "out" / "umap_grid.png").exists()
    assert ad.read_h5ad(prepared).obsm["X_pca"].shape[1] == 50
