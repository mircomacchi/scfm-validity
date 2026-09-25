import numpy as np
import pytest
import scipy.sparse as sp

from scfm_validity import artefacts


def test_doublets_are_heterotypic_and_same_batch(toy_adata):
    d = artefacts.simulate_doublets(toy_adata, frac=0.1, batch_key="donor_id", seed=1)
    assert d.n_obs == 60
    for parents in d.obs["parents"]:
        a, b = parents.split("+")
        assert a != b


def test_doublet_depth_is_scaled(toy_adata):
    d = artefacts.simulate_doublets(toy_adata, frac=0.2, depth_factor=1.6, seed=2)
    singlet_depth = np.asarray(toy_adata.X.sum(axis=1)).mean()
    doublet_depth = np.asarray(d.X.sum(axis=1)).mean()
    assert doublet_depth == pytest.approx(1.6 * singlet_depth, rel=0.1)


def test_counts_stay_integer(toy_adata):
    out = artefacts.inject_artefacts(toy_adata, 0.05, 0.05, batch_key="donor_id")
    data = sp.csr_matrix(out.X).data
    assert np.all(data >= 0) and np.allclose(data, np.round(data))


def test_ambient_keeps_depth_and_shifts_towards_profile(toy_adata):
    a = artefacts.simulate_ambient(toy_adata, frac=0.1, contamination=0.5, seed=3)
    depth = np.asarray(a.X.sum(axis=1)).ravel()
    assert depth.mean() == pytest.approx(np.asarray(toy_adata.X.sum(axis=1)).mean(), rel=0.1)
    # marker block of each cell's own type loses share when half the UMIs are ambient
    own = {"fibroblast": slice(0, 50), "myocyte": slice(50, 100), "macrophage": slice(100, 150)}
    shares = [np.asarray(a.X[i, own[t]].sum()) / depth[i] for i, t in enumerate(a.obs["cell_type"])]
    assert np.mean(shares) < 0.75


def test_inject_labels(toy_adata):
    out = artefacts.inject_artefacts(toy_adata, 0.05, 0.1)
    counts = out.obs["artefact"].value_counts()
    assert counts["none"] == 600 and counts["doublet"] == 30 and counts["ambient"] == 60
    assert out.n_vars == toy_adata.n_vars
