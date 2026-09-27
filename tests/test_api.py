import time

import pytest
from fastapi.testclient import TestClient

from scfm_validity import api


@pytest.fixture
def client():
    return TestClient(api.app)


def _wait(client, job_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_job_runs_and_returns_metrics(client, toy_adata, tmp_path):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)
    with path.open("rb") as fh:
        r = client.post(
            "/jobs",
            files={"file": ("toy.h5ad", fh, "application/octet-stream")},
            data={"methods": "pca", "n_cells": "600", "min_genes": "10"},
        )
    assert r.status_code == 202
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "done", body["error"]
    kinds = {row["artefact"] for row in body["result"]}
    assert kinds == {"doublet", "ambient"}
    assert all(0 <= row["detectability_auroc"] <= 1 for row in body["result"])


def test_rejects_non_h5ad(client):
    r = client.post("/jobs", files={"file": ("x.csv", b"a,b\n1,2", "text/csv")})
    assert r.status_code == 415


def test_rejects_geneformer_in_request(client, toy_adata, tmp_path):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)
    with path.open("rb") as fh:
        r = client.post(
            "/jobs",
            files={"file": ("toy.h5ad", fh, "application/octet-stream")},
            data={"methods": "geneformer"},
        )
    assert r.status_code == 422


def test_failed_job_reports_error(client, toy_adata, tmp_path):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)
    with path.open("rb") as fh:
        r = client.post(
            "/jobs",
            files={"file": ("toy.h5ad", fh, "application/octet-stream")},
            data={"label_key": "not_a_column", "n_cells": "600"},
        )
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "failed" and "not_a_column" in body["error"]


def test_unknown_job(client):
    assert client.get("/jobs/nope").status_code == 404
