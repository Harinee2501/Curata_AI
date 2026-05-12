# RL-Guided Synthetic Data Selection

Thompson-sampling bandit pipeline that decides which LLM-generated synthetic rows to keep — available in **two modes**: a Streamlit UI for end-users and a FastAPI REST API for developers.

---

## Project Structure

```
unisys_project/
├── api.py                 ← 🆕 FastAPI REST API
├── streamlit_app.py       ← 🆕 Improved Streamlit UI (v2)
├── pipeline.py            ← Core pipeline logic (shared by both)
├── config.py / config.yaml
├── bandit/thompson.py     ← Thompson Sampling bandit
├── classifier/            ← Logistic Regression trainer + evaluator
├── data/                  ← Dataset loader + LLM synthetic generator
├── features/              ← Uncertainty, verifier, novelty signals
├── outputs/               ← Generated output files land here
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt

# Copy and fill in your Groq API key
cp .env.example .env
# Add:  LLM_API_KEY=gsk_your_key_here
# Get a free key at https://console.groq.com
```

---

## Mode 1 — Streamlit UI (for general users)

No code needed — just a browser.

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**.

**Features:**
- Upload any labeled CSV (≥30 rows, numeric columns)
- Preview dataset + class distribution
- Choose label column and target synthetic row count
- Progress bar while pipeline runs
- Side-by-side F1 comparison: Baseline vs Naive Aug vs Bandit
- Download `curated.csv`, `augmented.csv`, `report.json`

---

## Mode 2 — FastAPI REST API (for developers)

Full REST API with auto-generated Swagger docs.

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Docs at **http://localhost:8000/docs**

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/jobs` | Submit async job |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/jobs/{job_id}/results` | Full results JSON |
| `GET` | `/jobs/{job_id}/download/{file}` | Download output file |
| `DELETE` | `/jobs/{job_id}` | Delete job + cleanup |
| `POST` | `/run-sync` | Synchronous run (small datasets) |

### Quick start with curl

```bash
# Submit
curl -X POST http://localhost:8000/jobs \
  -F "csv_file=@Iris.csv" \
  -F "label_col=Species" \
  -F "target_size=100"
# -> {"job_id": "abc-123", "status": "pending"}

# Poll
curl http://localhost:8000/jobs/abc-123
# -> {"status": "done"}

# Results
curl http://localhost:8000/jobs/abc-123/results

# Download
curl -O http://localhost:8000/jobs/abc-123/download/curated.csv
curl -O http://localhost:8000/jobs/abc-123/download/report.json
```

### Python integration example

```python
import requests, time

BASE = "http://localhost:8000"

resp = requests.post(f"{BASE}/jobs",
    files={"csv_file": open("data.csv", "rb")},
    data={"label_col": "label", "target_size": 200}
)
job_id = resp.json()["job_id"]

while True:
    status = requests.get(f"{BASE}/jobs/{job_id}").json()["status"]
    if status == "done": break
    if status == "error": raise RuntimeError("Pipeline failed")
    time.sleep(5)

results = requests.get(f"{BASE}/jobs/{job_id}/results").json()
print(f"Bandit F1: {results['bandit_test_f1']:.4f}")
```

---

## Pipeline Overview

```
User CSV → load → generate_synthetic_pool (LLM)
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
          Baseline    Naive Aug.   Thompson Bandit loop
          (real only) (all synth)    per batch:
                                       compute features
                                       accept / reject
                                       retrain model
                                       reward = ΔF1 > 0
                                            │
                                       Final model on
                                       real + curated
                                            │
                              save outputs (CSV + JSON)
```

---

## Output Files

| File | Contents |
|------|----------|
| `outputs/curated.csv` | Synthetic rows selected by the bandit |
| `outputs/augmented.csv` | Original + curated merged dataset |
| `outputs/report.json` | F1 scores + full bandit statistics |
