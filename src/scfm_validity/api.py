"""HTTP API: upload an h5ad, get an artefact-validity report.

    POST /jobs            multipart upload + parameters -> 202 {"job_id", "status"}
    GET  /jobs/{job_id}   status, and the metrics once the job has finished
    GET  /health          liveness probe

Jobs run in a worker thread so the request returns at once. The job store is in
memory: fine for one process, to be replaced by a queue and a database before
running several replicas.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from scfm_validity import __version__
from scfm_validity.pipeline import FAST_METHODS, run_report

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 2 * 1024**3
Status = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    status: Status = "queued"
    params: dict = field(default_factory=dict)
    result: list[dict] | None = None
    error: str | None = None


app = FastAPI(title="scfm-validity", version=__version__)
_jobs: dict[str, Job] = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1)  # one job at a time bounds peak memory


def _run(job_id: str, path: Path) -> None:
    with _lock:
        job = _jobs[job_id]
        job.status = "running"
    try:
        report = run_report(path, **job.params)
        with _lock:
            job.result = report.round(4).to_dict(orient="records")
            job.status = "done"
    except Exception as exc:  # reported to the client, not raised in the worker
        log.exception("Job %s failed", job_id)
        with _lock:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
    finally:
        path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(..., description="h5ad with raw counts in X or raw.X"),
    methods: str = Form("pca", description=f"Comma-separated subset of {FAST_METHODS}"),
    label_key: str = Form("cell_type"),
    batch_key: str = Form("donor_id"),
    n_cells: int = Form(20_000, ge=100, le=100_000),
    min_genes: int = Form(200, ge=0),
    seed: int = Form(0),
) -> dict:
    if not (file.filename or "").endswith(".h5ad"):
        raise HTTPException(415, "Upload an .h5ad file")
    chosen = tuple(m.strip() for m in methods.split(",") if m.strip())
    if not chosen or set(chosen) - set(FAST_METHODS):
        raise HTTPException(422, f"methods must be a subset of {list(FAST_METHODS)}")

    tmp = Path(tempfile.mkstemp(suffix=".h5ad")[1])
    size = 0
    with tmp.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, "File larger than 2 GB")
            out.write(chunk)

    job_id = uuid.uuid4().hex
    params = dict(
        methods=chosen,
        label_key=label_key,
        batch_key=batch_key or None,
        n_cells=n_cells,
        min_genes=min_genes,
        seed=seed,
    )
    with _lock:
        _jobs[job_id] = Job(params=params)
    _executor.submit(_run, job_id, tmp)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return {
            "job_id": job_id,
            "status": job.status,
            "params": job.params,
            "result": job.result,
            "error": job.error,
        }
