"""Three ways to embed the same cells: a PCA baseline, a trained scVI model, and
Geneformer used zero-shot through the `helical` package.

Every function takes raw counts in `.X`, writes its embedding to `adata.obsm`
and returns the key it used.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc

log = logging.getLogger(__name__)


def embed_pca(adata: ad.AnnData, n_hvg: int = 2000, n_comps: int = 50, seed: int = 0) -> str:
    """Log-normalise, select highly variable genes and run PCA."""
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat")
    sc.pp.pca(tmp, n_comps=n_comps, mask_var="highly_variable", random_state=seed)
    adata.obsm["X_pca"] = tmp.obsm["X_pca"]
    return "X_pca"


def embed_scvi(
    adata: ad.AnnData,
    batch_key: str | None,
    n_hvg: int = 2000,
    n_latent: int = 30,
    max_epochs: int = 100,
    seed: int = 0,
) -> str:
    """Train an scVI model on raw counts of the highly variable genes.

    scVI models counts with a zero-inflated negative binomial likelihood and
    conditions on `batch_key`, so the latent space is corrected for batch.
    """
    import scvi

    scvi.settings.seed = seed
    tmp = adata.copy()
    sc.pp.highly_variable_genes(
        tmp, n_top_genes=n_hvg, flavor="seurat_v3", batch_key=batch_key, subset=True
    )
    scvi.model.SCVI.setup_anndata(tmp, batch_key=batch_key)
    model = scvi.model.SCVI(tmp, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True, enable_progress_bar=False)
    adata.obsm["X_scvi"] = model.get_latent_representation()
    adata.uns["scvi_history"] = {
        k: v.to_numpy().ravel().tolist() for k, v in model.history.items() if "elbo" in k
    }
    return "X_scvi"


def embed_geneformer(
    adata: ad.AnnData,
    model_name: str = "gf-12L-38M-i4096",
    batch_size: int = 16,
    device: str | None = None,
) -> str:
    """Zero-shot Geneformer cell embeddings via `helical`.

    Geneformer ranks each cell's genes by expression (normalised by each gene's
    median across its pretraining corpus) and reads the rank list with a
    transformer. No fine-tuning here: the point is what the pretrained model
    already encodes.
    """
    import torch
    from helical.models.geneformer import Geneformer, GeneformerConfig

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = GeneformerConfig(model_name=model_name, batch_size=batch_size, device=device)
    model = Geneformer(configurer=config)
    dataset = model.process_data(adata, gene_names="ensembl_id")
    Z = np.asarray(model.get_embeddings(dataset))
    key = "X_geneformer"
    adata.obsm[key] = Z
    log.info("Geneformer %s on %s: %s", model_name, device, Z.shape)
    return key
