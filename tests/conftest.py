import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def toy_adata() -> ad.AnnData:
    """600 cells, 300 genes, three cell types with distinct marker blocks, two batches."""
    rng = np.random.default_rng(0)
    n_per, n_genes = 200, 300
    types = np.repeat(["fibroblast", "myocyte", "macrophage"], n_per)
    base = rng.gamma(0.3, 1.0, size=n_genes)
    X = []
    for i, _ in enumerate(["fibroblast", "myocyte", "macrophage"]):
        mu = base.copy()
        mu[i * 50 : (i + 1) * 50] *= 20
        X.append(rng.poisson(mu, size=(n_per, n_genes)))
    X = sp.csr_matrix(np.vstack(X), dtype=np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(types),
            "donor_id": pd.Categorical(np.tile(["d1", "d2"], 300)),
        },
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(index=[f"ENSG{i:011d}" for i in range(n_genes)])
    var["feature_name"] = [f"GENE{i}" for i in range(n_genes)]
    var["ensembl_id"] = var.index
    return ad.AnnData(X=X, obs=obs, var=var)
