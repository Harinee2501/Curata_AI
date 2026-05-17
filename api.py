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
from typing import Literal, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="RL-Guided Synthetic Data Selection API",
    description=(
        "Upload a labeled CSV, run the RL-guided or Thompson-bandit curation pipeline, "
        "and download curated synthetic datasets. All endpoints are stateless "
        "except for the job-tracking store which lives in memory.\n\n"
        "### Augmentation modes\n"
        "| Mode | Pool size | RL steps | Patience | Threshold | Est. time |\n"
        "|------|-----------|----------|----------|-----------|-----------|\n"
        "| `fast` | 1× dataset | 10 | 3 | 0.50 | ~1–2 min |\n"
        "| `balanced` | 3× dataset | 40 | 8 | 0.65 | ~5–10 min |\n"
        "| `thorough` | 5× dataset | 100 | 20 | 0.75 | ~20–40 min |\n\n"
        "### Classifiers\n"
        "| Key | Description | Notes |\n"
        "|-----|-------------|-------|\n"
        "| `logistic_regression` | Fast linear baseline | Default |\n"
        "| `naive_bayes` | Very fast; good for high-dimensional data | — |\n"
        "| `random_forest` | Robust ensemble; handles non-linearity | — |\n"
        "| `xgboost` | Gradient boosting (XGBoost) | `pip install xgboost` |\n"
        "| `lightgbm` | Gradient boosting (LightGBM) | `pip install lightgbm` |\n"
        "| `svm` | RBF kernel SVM; strong on small datasets | — |\n"
    ),
    version="2.0.0",
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
# Maps job_id -> {"status": ..., "results": ..., "output_dir": ..., "message": ...}
JOB_STORE: dict[str, dict] = {}

# Valid augmentation modes (mirrors config.py VALID_MODES)
_VALID_MODES = {"fast", "balanced", "thorough"}

# Valid classifiers (mirrors config.py VALID_CLASSIFIERS)
_VALID_CLASSIFIERS = {
    "logistic_regression",
    "naive_bayes",
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm",
}


# ── Pydantic schemas ─────────────────────────────────────────────────
class JobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="pending | running | done | error")
    message: Optional[str] = None


class PipelineResults(BaseModel):
    job_id: str
    status: str
    augmentation_mode: Optional[str] = None
    classifier_model: Optional[str] = None
    dataset: Optional[str] = None
    baseline_test_f1: Optional[float] = None
    naive_test_f1: Optional[float] = None
    bandit_test_f1: Optional[float] = None
    curated_size: Optional[int] = None
    acceptance_rate: Optional[float] = None
    bandit_summary: Optional[dict] = None
    full_report: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────

def _validate_mode(augmentation_mode: str) -> str:
    mode = augmentation_mode.lower().strip()
    if mode not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"augmentation_mode must be one of {sorted(_VALID_MODES)}, got '{augmentation_mode}'.",
        )
    return mode


def _validate_classifier(classifier_model: str) -> str:
    clf = classifier_model.lower().strip()
    if clf not in _VALID_CLASSIFIERS:
        raise HTTPException(
            status_code=422,
            detail=f"classifier_model must be one of {sorted(_VALID_CLASSIFIERS)}, got '{classifier_model}'.",
        )
    return clf


# ── Background task ───────────────────────────────────────────────────
def _run_pipeline_task(
    job_id: str,
    csv_path: str,
    label_col: str,
    augmentation_mode: str,
    classifier_model: str,
    output_dir: str,
    text_col: str | None = None,
):
    """Runs in a BackgroundTask so the POST returns immediately."""
    import sys, os

    sys.path.insert(0, os.path.dirname(__file__))

    JOB_STORE[job_id]["status"] = "running"
    try:
        from config import cfg

        original_output = cfg.OUTPUT_DIR
        cfg.OUTPUT_DIR = output_dir

        from pipeline import run_pipeline

        results = run_pipeline(
            csv_path=csv_path,
            label_col=label_col,
            augmentation_mode=augmentation_mode,
            classifier_model=classifier_model,
            text_col=text_col,
        )

        cfg.OUTPUT_DIR = original_output

        JOB_STORE[job_id]["status"] = "done"
        JOB_STORE[job_id]["results"] = results
    except Exception as exc:
        JOB_STORE[job_id]["status"] = "error"
        JOB_STORE[job_id]["message"] = str(exc)
    finally:
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


@app.get("/modes", tags=["Pipeline"])
def list_modes():
    """
    **List available augmentation modes** and their configuration.

    Use one of these mode names in the `augmentation_mode` field when
    submitting a job.
    """
    from config import cfg

    return {mode: cfg.get_mode_config(mode) for mode in sorted(_VALID_MODES)}


@app.get("/classifiers", tags=["Pipeline"])
def list_classifiers():
    """
    **List available classifier models** with descriptions and install notes.

    Use one of these keys in the `classifier_model` field when submitting a job.
    """
    return {
        "logistic_regression": {
            "description": "Fast linear baseline; well-calibrated probabilities.",
            "requires_install": None,
        },
        "naive_bayes": {
            "description": "Very fast; works well on high-dimensional or sparse data.",
            "requires_install": None,
        },
        "random_forest": {
            "description": "Robust ensemble; handles non-linearity and noisy features.",
            "requires_install": None,
        },
        "xgboost": {
            "description": "Gradient boosting via XGBoost; strong on tabular data.",
            "requires_install": "pip install xgboost",
        },
        "lightgbm": {
            "description": "Gradient boosting via LightGBM; fast on large datasets.",
            "requires_install": "pip install lightgbm",
        },
        "svm": {
            "description": "RBF kernel SVM with Platt scaling; strong on small datasets.",
            "requires_install": None,
        },
    }


@app.post("/jobs", response_model=JobStatus, status_code=202, tags=["Pipeline"])
async def submit_job(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(..., description="Labeled CSV dataset"),
    label_col: str = Form(
        ..., description="Name of the label/target column in the CSV"
    ),
    augmentation_mode: str = Form(
        "balanced",
        description=(
            "How aggressively to augment. One of: `fast`, `balanced`, `thorough`. "
            "Controls pool size, RL steps, early-stopping patience, and curation "
            "threshold together. Defaults to `balanced`."
        ),
    ),
    classifier_model: str = Form(
        "logistic_regression",
        description=(
            "Classifier to train and evaluate on. One of: `logistic_regression`, "
            "`naive_bayes`, `random_forest`, `xgboost`, `lightgbm`, `svm`. "
            "Defaults to `logistic_regression`."
        ),
    ),
    text_col: Optional[str] = Form(
        None,
        description="Optional document column for text classification (e.g. 'review'). "
        "Leave empty to auto-detect when there is a single non-numeric feature column.",
    ),
):
    """
    **Submit a pipeline job** (async).

    Upload your CSV, choose an augmentation mode, and pick a classifier.
    The pipeline runs in the background — returns a `job_id` immediately.
    Poll `/jobs/{job_id}` for status.

    ### Workflow
    1. POST to `/jobs` → get `job_id`
    2. GET `/jobs/{job_id}` → wait for `status: done`
    3. GET `/jobs/{job_id}/results` → full metrics
    4. GET `/jobs/{job_id}/download/{filename}` → download output files

    ### Augmentation modes
    - **fast** — 1× pool, 10 RL steps, loose threshold (~1–2 min)
    - **balanced** — 3× pool, 40 RL steps, balanced threshold (~5–10 min)
    - **thorough** — 5× pool, 100 RL steps, strict threshold (~20–40 min)

    ### Classifiers
    - **logistic_regression** — fast linear baseline (default)
    - **naive_bayes** — very fast; good for sparse/high-dimensional data
    - **random_forest** — robust ensemble
    - **xgboost** — gradient boosting (`pip install xgboost` required)
    - **lightgbm** — fast gradient boosting (`pip install lightgbm` required)
    - **svm** — RBF kernel SVM; strong on small datasets

    ### Available output files
    - `curated.csv` — bandit-selected synthetic samples
    - `augmented.csv` — original + curated data merged
    - `report.json` — full evaluation report
    """
    mode = _validate_mode(augmentation_mode)
    clf = _validate_classifier(classifier_model)

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
        "augmentation_mode": mode,
        "classifier_model": clf,
    }

    background_tasks.add_task(
        _run_pipeline_task,
        job_id=job_id,
        csv_path=tmp_path,
        label_col=label_col,
        augmentation_mode=mode,
        classifier_model=clf,
        output_dir=output_dir,
        text_col=text_col.strip() if text_col and text_col.strip() else None,
    )

    return JobStatus(
        job_id=job_id,
        status="pending",
        message=f"Job accepted (mode={mode}, classifier={clf}). Poll /jobs/{job_id} for status.",
    )


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
        raise HTTPException(
            status_code=202, detail="Job is pending. Try again shortly."
        )
    if job["status"] == "running":
        raise HTTPException(
            status_code=202, detail="Job is still running. Try again shortly."
        )
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
        augmentation_mode=job.get("augmentation_mode") or r.get("augmentation_mode"),
        classifier_model=job.get("classifier_model") or r.get("classifier_model"),
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
        raise HTTPException(
            status_code=400, detail=f"Filename must be one of {ALLOWED}"
        )

    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "done":
        raise HTTPException(
            status_code=409, detail=f"Job is not done yet (status: {job['status']})."
        )

    file_path = Path(job["output_dir"]) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Output file '{filename}' not found."
        )

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
    augmentation_mode: str = Form(
        "balanced",
        description="One of: `fast`, `balanced`, `thorough`. Defaults to `balanced`.",
    ),
    classifier_model: str = Form(
        "logistic_regression",
        description=(
            "One of: `logistic_regression`, `naive_bayes`, `random_forest`, "
            "`xgboost`, `lightgbm`, `svm`. Defaults to `logistic_regression`."
        ),
    ),
    text_col: Optional[str] = Form(
        None, description="Optional text/document column name"
    ),
):
    """
    **Run the pipeline synchronously** and return results immediately.

    ⚠️ Blocks until done — suitable for small datasets or quick tests.
    Use `/jobs` (async) for larger datasets to avoid timeout.
    """
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    mode = _validate_mode(augmentation_mode)
    clf = _validate_classifier(classifier_model)

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
            augmentation_mode=mode,
            classifier_model=clf,
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
        {
            "job_id": jid,
            "status": job["status"],
            "augmentation_mode": job.get("augmentation_mode"),
            "classifier_model": job.get("classifier_model"),
        }
        for jid, job in JOB_STORE.items()
    ]