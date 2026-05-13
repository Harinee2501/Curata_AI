# api.py
# ─────────────────────────────────────────────────────────────────────
# FastAPI REST API — exposes the RL-guided synthetic data pipeline
# so developers can integrate it programmatically.
#
# Run:
#   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
#
# Docs auto-generated at:
#   http://localhost:8000/docs      (Swagger UI)
#   http://localhost:8000/redoc     (ReDoc)
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="RL-Guided Synthetic Data Selection API",
    description=(
        "Upload a labeled CSV, run the Thompson-sampling bandit pipeline, "
        "and download curated synthetic datasets.  All endpoints are stateless "
        "except for the job-tracking store which lives in memory."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────
# Maps job_id -> {"status": ..., "results": ..., "output_dir": ...}
JOB_STORE: dict[str, dict] = {}


# ── Pydantic schemas ─────────────────────────────────────────────────
class JobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="pending | running | done | error")
    message: Optional[str] = None


class PipelineResults(BaseModel):
    job_id: str
    status: str
    dataset: Optional[str] = None
    baseline_test_f1: Optional[float] = None
    naive_test_f1: Optional[float] = None
    bandit_test_f1: Optional[float] = None
    curated_size: Optional[int] = None
    acceptance_rate: Optional[float] = None
    bandit_summary: Optional[dict] = None
    full_report: Optional[dict] = None


# ── Background task ───────────────────────────────────────────────────
def _run_pipeline_task(
    job_id: str,
    csv_path: str,
    label_col: str,
    target_size: int,
    output_dir: str,
    text_col: str | None = None,
):
    """Runs in a BackgroundTask so the POST returns immediately."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    JOB_STORE[job_id]["status"] = "running"
    try:
        from config import cfg
        # Override output dir per-job so jobs don't collide
        original_output = cfg.OUTPUT_DIR
        cfg.OUTPUT_DIR = output_dir

        from pipeline import run_pipeline
        results = run_pipeline(
            csv_path=csv_path,
            label_col=label_col,
            target_size=target_size,
            text_col=text_col,
        )

        cfg.OUTPUT_DIR = original_output  # restore

        JOB_STORE[job_id]["status"] = "done"
        JOB_STORE[job_id]["results"] = results
    except Exception as exc:
        JOB_STORE[job_id]["status"] = "error"
        JOB_STORE[job_id]["message"] = str(exc)
    finally:
        # clean up uploaded CSV
        try:
            os.unlink(csv_path)
        except Exception:
            pass


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "RL Synthetic Data Selection API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/jobs", response_model=JobStatus, status_code=202, tags=["Pipeline"])
async def submit_job(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(..., description="Labeled CSV dataset"),
    label_col: str = Form(..., description="Name of the label/target column in the CSV"),
    target_size: int = Form(200, description="Number of curated synthetic rows to select", ge=1, le=10000),
    text_col: Optional[str] = Form(
        None,
        description="Optional document column for text classification (e.g. review). "
        "Leave empty to auto-detect when there is a single non-numeric feature column.",
    ),
):
    """
    **Submit a pipeline job** (async).

    Upload your CSV and configuration. The pipeline runs in the background.
    Returns a `job_id` immediately — poll `/jobs/{job_id}` for status.

    ### Workflow
    1. POST to `/jobs` → get `job_id`
    2. GET `/jobs/{job_id}` → wait for `status: done`
    3. GET `/jobs/{job_id}/results` → full metrics
    4. GET `/jobs/{job_id}/download/{filename}` → download output files

    ### Available output files
    - `curated.csv` — bandit-selected synthetic samples
    - `augmented.csv` — original + curated data merged
    - `report.json` — full evaluation report
    """
    # Save uploaded file to a temp location
    suffix = Path(csv_file.filename).suffix or ".csv"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as f:
        content = await csv_file.read()
        f.write(content)

    # Create a per-job output directory
    job_id = str(uuid.uuid4())
    output_dir = str(Path(tempfile.mkdtemp()) / job_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    JOB_STORE[job_id] = {
        "status": "pending",
        "results": None,
        "output_dir": output_dir,
        "message": None,
    }

    background_tasks.add_task(
        _run_pipeline_task,
        job_id=job_id,
        csv_path=tmp_path,
        label_col=label_col,
        target_size=target_size,
        output_dir=output_dir,
        text_col=text_col.strip() if text_col and text_col.strip() else None,
    )

    return JobStatus(job_id=job_id, status="pending", message="Job accepted. Poll /jobs/{job_id} for status.")


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Pipeline"])
def get_job_status(job_id: str):
    """
    **Poll job status.**

    Returns `pending`, `running`, `done`, or `error`.
    """
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
    )


@app.get("/jobs/{job_id}/results", response_model=PipelineResults, tags=["Pipeline"])
def get_job_results(job_id: str):
    """
    **Retrieve full results** once the job is done.

    Raises 202 if still running, 500 if errored.
    """
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] == "pending":
        raise HTTPException(status_code=202, detail="Job is pending. Try again shortly.")
    if job["status"] == "running":
        raise HTTPException(status_code=202, detail="Job is still running. Try again shortly.")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('message')}")

    r = job["results"] or {}
    bandit = r.get("bandit_guided", {})
    baseline = r.get("baseline_real_only", {})
    naive = r.get("naive_augmentation", {})
    summary = bandit.get("bandit_summary", {})

    curated_size: Optional[int] = None
    rl_meta = r.get("rl_guided") or {}
    if rl_meta.get("curated_rows") is not None:
        try:
            curated_size = int(rl_meta["curated_rows"])
        except (TypeError, ValueError):
            curated_size = None
    if curated_size is None:
        try:
            curated_path = Path(job["output_dir"]) / "curated.csv"
            if curated_path.exists():
                curated_size = int(pd.read_csv(curated_path).shape[0])
        except Exception:
            curated_size = None

    return PipelineResults(
        job_id=job_id,
        status="done",
        dataset=r.get("dataset"),
        baseline_test_f1=baseline.get("test_f1"),
        naive_test_f1=naive.get("test_f1"),
        bandit_test_f1=bandit.get("test_f1"),
        curated_size=curated_size,
        acceptance_rate=summary.get("acceptance_rate"),
        bandit_summary=summary,
        full_report=r,
    )


@app.get("/jobs/{job_id}/download/{filename}", tags=["Pipeline"])
def download_output(job_id: str, filename: str):
    """
    **Download output files** after a job completes.

    Available filenames:
    - `curated.csv`
    - `augmented.csv`
    - `report.json`
    """
    ALLOWED = {"curated.csv", "augmented.csv", "report.json"}
    if filename not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Filename must be one of {ALLOWED}")

    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done yet (status: {job['status']}).")

    file_path = Path(job["output_dir"]) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found.")

    media = "text/csv" if filename.endswith(".csv") else "application/json"
    return FileResponse(str(file_path), media_type=media, filename=filename)


@app.delete("/jobs/{job_id}", tags=["Pipeline"])
def delete_job(job_id: str):
    """
    **Delete a job** and clean up its output files.
    """
    job = JOB_STORE.pop(job_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    try:
        shutil.rmtree(job["output_dir"], ignore_errors=True)
    except Exception:
        pass
    return {"message": f"Job '{job_id}' deleted."}


@app.post("/run-sync", tags=["Pipeline (Sync)"])
async def run_sync(
    csv_file: UploadFile = File(..., description="Labeled CSV dataset"),
    label_col: str = Form(..., description="Name of the label/target column"),
    target_size: int = Form(200, ge=1, le=2000),
    text_col: Optional[str] = Form(None, description="Optional text/document column name"),
):
    """
    **Run the pipeline synchronously** and return results immediately.

    ⚠️ Blocks until done — suitable for small datasets or quick tests.
    Use `/jobs` (async) for larger datasets to avoid timeout.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    suffix = Path(csv_file.filename).suffix or ".csv"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    output_dir = tempfile.mkdtemp()

    try:
        with os.fdopen(tmp_fd, "wb") as f:
            content = await csv_file.read()
            f.write(content)

        from config import cfg
        original_output = cfg.OUTPUT_DIR
        cfg.OUTPUT_DIR = output_dir

        from pipeline import run_pipeline
        results = run_pipeline(
            csv_path=tmp_path,
            label_col=label_col,
            target_size=target_size,
            text_col=text_col.strip() if text_col and text_col.strip() else None,
        )

        cfg.OUTPUT_DIR = original_output
        return JSONResponse(content=results)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass


@app.get("/jobs", tags=["Pipeline"])
def list_jobs():
    """
    **List all jobs** and their current status.
    """
    return [
        {"job_id": jid, "status": job["status"]}
        for jid, job in JOB_STORE.items()
    ]
