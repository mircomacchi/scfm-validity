"""Three ways to embed the same cells: a PCA baseline, a trained scVI model, and
Geneformer used zero-shot through the `helical` package.

Every function takes raw counts in `.X`, writes its embedding to `adata.obsm`
and returns the key it used.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

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


def vocab_subset(adata: ad.AnnData, gene_token_dict: dict) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Counts restricted to one column per in-vocabulary Ensembl ID.

    Used with `tk.collapse_gene_ids = False`: helical 3.1.3's own collapsing drops
    var["ensembl_id"] and then fails (helicalAI/helical#433).
    """
    ids = adata.var["ensembl_id"].astype(str).str.upper()
    keep = ids.isin(gene_token_dict.keys()).to_numpy() & ~ids.duplicated().to_numpy()
    X = sp.csr_matrix(adata.X)[:, np.flatnonzero(keep)]
    var = pd.DataFrame({"ensembl_id": ids[keep].to_numpy()}, index=ids[keep].to_numpy())
    log.info("Geneformer vocabulary covers %d of %d genes", keep.sum(), adata.n_vars)
    return X, var


def embed_geneformer(
    adata: ad.AnnData,
    model_name: str = "gf-12L-38M-i4096",
    batch_size: int = 4,
    device: str | None = None,
    chunk_size: int = 1000,
    checkpoint_dir: str | Path | None = None,
    max_chunks: int | None = None,
    weights: str | Path | None = None,
    key: str = "X_geneformer",
) -> str | None:
    """Geneformer cell embeddings via `helical`, zero-shot or from fine-tuned weights.

    Geneformer ranks each cell's genes by expression (normalised by each gene's
    median across its pretraining corpus) and reads the rank list with a
    transformer. With `weights` (a directory written by `finetune.finetune_geneformer`)
    the fine-tuned backbone is used instead of the pretrained one.

    Cells are tokenised and embedded `chunk_size` at a time, so peak memory holds
    one chunk of tokens rather than the whole dataset. With `checkpoint_dir`, each
    chunk is saved as it finishes and a rerun skips the chunks already on disk.
    With `max_chunks`, stop after computing that many new chunks and return None
    if the embedding is still incomplete. helical/PyTorch do not release all MPS
    memory between chunks (footprint reached 40 GB over 11 chunks on a 16 GB Mac),
    so long runs should compute one chunk per process: see `hpc/geneformer_local.sh`.
    """
    import gc

    import torch
    from helical.models.geneformer import Geneformer, GeneformerConfig

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if weights is not None:
        from scfm_validity.finetune import load_finetuned

        model = load_finetuned(weights, device=device, batch_size=batch_size)
        model_name = model.config["model_name"] if "model_name" in model.config else model_name
    else:
        config = GeneformerConfig(model_name=model_name, batch_size=batch_size, device=device)
        model = Geneformer(configurer=config)
    model.tk.collapse_gene_ids = False
    X, var = vocab_subset(adata, model.tk.gene_token_dict)

    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)
    parts, computed = [], 0
    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        done = ckpt / f"{key}_{start:07d}_{stop:07d}.npy" if ckpt else None
        if done and done.exists():
            parts.append(np.load(done))
            continue
        if max_chunks is not None and computed >= max_chunks:
            log.info("Stopping after %d new chunk(s); rerun to continue", computed)
            return None
        chunk = ad.AnnData(
            X=X[start:stop], obs=pd.DataFrame(index=adata.obs_names[start:stop]), var=var
        )
        dataset = model.process_data(chunk, gene_names="ensembl_id")
        parts.append(np.asarray(model.get_embeddings(dataset), dtype=np.float32))
        if done:
            np.save(done, parts[-1])
        computed += 1
        del chunk, dataset
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        log.info("%s: %d / %d cells", key, stop, adata.n_obs)

    adata.obsm[key] = np.vstack(parts)
    log.info("%s (%s on %s): %s", key, model_name, device, adata.obsm[key].shape)
    return key
