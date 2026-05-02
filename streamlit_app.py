import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import cfg


st.set_page_config(page_title="RL-Guided Synthetic Data Selection", layout="wide")


def _safe_filename(name: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name)
    return s or "uploaded.csv"


def _summarize_results(results: dict) -> dict:
    baseline = results.get("baseline_real_only", {})
    naive = results.get("naive_augmentation", {})
    bandit = results.get("bandit_guided", {})
    return {
        "dataset": results.get("dataset"),
        "label_col": results.get("label_col"),
        "num_classes": results.get("num_classes"),
        "target_size": results.get("target_size"),
        "curated_size": results.get("curated_size"),
        "baseline_test_f1": baseline.get("test_f1"),
        "naive_test_f1": naive.get("test_f1"),
        "bandit_test_f1": bandit.get("test_f1"),
        "acceptance_rate": (bandit.get("bandit_summary") or {}).get("acceptance_rate"),
        "llm_model": results.get("llm_model"),
    }


st.title("RL-Guided Synthetic Data Selection")
st.caption("Upload a labeled CSV, choose the label column and target synthetic rows, then run bandit-guided curation.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    target = st.number_input("Target curated synthetic rows", min_value=1, value=200, step=10)

    st.divider()
    st.subheader("Config (read-only)")
    st.write(
        {
            "classifier_model": cfg.CLASSIFIER_MODEL,
            "embedding_model": cfg.EMBEDDING_MODEL,
            "synth_pool_multiplier": cfg.SYNTH_POOL_MULTIPLIER,
            "synth_batch_size": cfg.SYNTH_BATCH_SIZE,
            "num_iterations": cfg.NUM_ITERATIONS,
            "output_dir": cfg.OUTPUT_DIR,
        }
    )

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df_preview = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

cols = list(df_preview.columns)
if not cols:
    st.error("CSV has no columns.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("Dataset preview")
    st.dataframe(df_preview.head(25), use_container_width=True)

with right:
    st.subheader("Run settings")
    label_col = st.selectbox("Label column", options=cols, index=max(0, len(cols) - 1))

    run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

if not run_clicked:
    st.stop()

run_root = Path(cfg.OUTPUT_DIR)
run_root.mkdir(parents=True, exist_ok=True)

with st.spinner("Running pipeline (this can take a while)..."):
    try:
        # Import lazily so the app loads fast (pipeline imports torch/transformers).
        from pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            csv_path = tmpdir_path / _safe_filename(uploaded.name)
            with open(csv_path, "wb") as f:
                f.write(uploaded.getbuffer())

            results = run_pipeline(
                csv_path=str(csv_path),
                label_col=str(label_col),
                target_size=int(target),
            )

    except EnvironmentError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

st.success("Done.")

st.subheader("Summary")
st.json(_summarize_results(results))

st.subheader("Full evaluation report")
st.json(results)

out_dir = Path(cfg.OUTPUT_DIR)
outt_path = out_dir / "outt.csv"
report_path = out_dir / "experiment_report.json"

download_cols = st.columns([1, 1, 2])
with download_cols[0]:
    if outt_path.exists():
        st.download_button(
            "Download outt.csv",
            data=outt_path.read_bytes(),
            file_name="outt.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("outt.csv not found in outputs.")

with download_cols[1]:
    if report_path.exists():
        st.download_button(
            "Download experiment_report.json",
            data=report_path.read_bytes(),
            file_name="experiment_report.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("experiment_report.json not found in outputs.")

st.subheader("Output files")
st.write(
    {
        "outputs_dir": str(out_dir.resolve()),
        "outt.csv": str(outt_path.resolve()) if outt_path.exists() else None,
        "curated_synthetic_samples.csv": str((out_dir / "curated_synthetic_samples.csv").resolve())
        if (out_dir / "curated_synthetic_samples.csv").exists()
        else None,
        "augmented_dataset.csv": str((out_dir / "augmented_dataset.csv").resolve())
        if (out_dir / "augmented_dataset.csv").exists()
        else None,
        "experiment_report.json": str(report_path.resolve()) if report_path.exists() else None,
        "pipeline.log": str((out_dir / "pipeline.log").resolve()) if (out_dir / "pipeline.log").exists() else None,
    }
)

