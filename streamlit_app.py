# streamlit_app.py  (v2 — improved UX)
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import cfg

st.set_page_config(
    page_title="RL Synthetic Data Selection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card{background:#1e1e2e;border-radius:10px;padding:16px;margin:4px 0;border:1px solid #313244;}
.metric-label{font-size:.8rem;color:#a6adc8;margin-bottom:4px;}
.metric-value{font-size:1.8rem;font-weight:700;color:#cdd6f4;}
.metric-delta{font-size:.75rem;margin-top:2px;}
.delta-pos{color:#a6e3a1;}
.delta-neg{color:#f38ba8;}
</style>
""", unsafe_allow_html=True)


def _safe_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name) or "upload.csv"


def _f1_card(label: str, value, ref=None) -> str:
    if value is None:
        return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">—</div></div>'
    delta_html = ""
    if ref is not None and ref > 0:
        delta = value - ref
        cls = "delta-pos" if delta >= 0 else "delta-neg"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="metric-delta {cls}">{sign}{delta:.4f} vs baseline</div>'
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value:.4f}</div>{delta_html}</div>'
    )


with st.sidebar:
    st.title("🤖 RL Synthetic Data\nSelection")
    st.caption("Thompson Sampling · Tabular or text (sentence-embedding) features")
    st.divider()
    st.header("⚙️ Inputs")
    uploaded = st.file_uploader("Upload labeled CSV", type=["csv"],
                                 help="Must have ≥30 rows and a clear label column.")
    target = st.slider("Target synthetic rows", 10, 500, 200, 10,
                        help="How many bandit-curated synthetic rows to select.")
    st.divider()
    st.subheader("🔧 Config")
    st.json({
        "classifier": cfg.CLASSIFIER_MODEL,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "synth_pool_multiplier": cfg.SYNTH_POOL_MULTIPLIER,
        "random_seed": cfg.RANDOM_SEED,
    }, expanded=False)
    st.divider()
    st.caption("💡 Need programmatic access?\nUse the **REST API**:\n```\nuvicorn api:app --reload\n```")


st.title("🤖 RL-Guided Synthetic Data Selection")
st.caption("Upload a labeled CSV → Thompson-sampling bandit curates LLM-generated synthetic rows → compare methods.")

if uploaded is None:
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1** — Upload a labeled CSV in the sidebar.")
    c2.info("**Step 2** — Select the label column and target size.")
    c3.info("**Step 3** — Click **Run Pipeline** and wait for results.")
    st.stop()

try:
    df_preview = pd.read_csv(uploaded)
    uploaded.seek(0)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

cols = list(df_preview.columns)
if not cols:
    st.error("CSV has no columns.")
    st.stop()

tab_data, tab_run, tab_results = st.tabs(["📊 Data Preview", "🚀 Run Pipeline", "📈 Results"])

with tab_data:
    st.subheader(f"Dataset — {len(df_preview):,} rows × {len(cols)} columns")
    left, right = st.columns([3, 1])
    with left:
        st.dataframe(df_preview.head(30), use_container_width=True, height=420)
    with right:
        st.metric("Rows", f"{len(df_preview):,}")
        st.metric("Columns", len(cols))
        st.subheader("Column types")
        st.dataframe(pd.DataFrame({"Column": df_preview.dtypes.index,
                                    "Type": df_preview.dtypes.astype(str).values}),
                     hide_index=True, use_container_width=True)
        nulls = df_preview.isnull().sum().sum()
        if nulls:
            st.warning(f"⚠️ {nulls} null values — will be dropped.")

with tab_run:
    st.subheader("Configure and run")
    rc1, rc2 = st.columns(2)
    with rc1:
        label_col = st.selectbox("Label column", options=cols,
                                  index=max(0, len(cols) - 1),
                                  help="Column containing your class labels.")
        if label_col:
            unique_labels = df_preview[label_col].dropna().unique()
            st.write(f"**Classes:** {sorted(str(v) for v in unique_labels)}")
    with rc2:
        text_choices = ["(auto)"] + [c for c in cols if c != label_col]
        text_col_ui = st.selectbox(
            "Text column (optional)",
            options=text_choices,
            index=0,
            help="IMDB-style: choose the review column, or auto when there is exactly one non-numeric feature column.",
        )
        st.metric("Target synthetic rows", target)
        st.metric("Available rows", len(df_preview))
        if len(df_preview) < cfg.MIN_REAL_SAMPLES:
            st.error(f"❌ Need ≥{cfg.MIN_REAL_SAMPLES} rows (have {len(df_preview)}).")

    st.divider()

    if not cfg.LLM_API_KEY:
        st.warning("⚠️ **LLM_API_KEY not set.** Add your Groq key to `.env`:  \n```\nLLM_API_KEY=gsk_...\n```\nGet a free key at [console.groq.com](https://console.groq.com).")

    run_clicked = st.button(
        "▶ Run Pipeline", type="primary", use_container_width=True,
        disabled=(len(df_preview) < cfg.MIN_REAL_SAMPLES),
    )

    if not run_clicked:
        st.stop()

    progress_bar = st.progress(0, text="Initialising…")
    status_box = st.empty()

    try:
        from pipeline import run_pipeline  # lazy import — torch/transformers are heavy
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / _safe_filename(uploaded.name)
            csv_path.write_bytes(uploaded.getbuffer())
            progress_bar.progress(10, text="Dataset loaded…")
            status_box.info("🔄 Running — this can take a few minutes.")
            results = run_pipeline(
                csv_path=str(csv_path),
                label_col=str(label_col),
                target_size=int(target),
                text_col=None if text_col_ui == "(auto)" else str(text_col_ui),
            )
        progress_bar.progress(100, text="Done!")
        st.session_state["results"] = results
        status_box.success("✅ Pipeline complete! See the **Results** tab.")
    except EnvironmentError as e:
        progress_bar.empty()
        st.error(str(e))
        st.stop()
    except Exception as e:
        progress_bar.empty()
        st.exception(e)
        st.stop()

with tab_results:
    results = st.session_state.get("results")
    if results is None:
        st.info("Run the pipeline first (in the **Run Pipeline** tab).")
        st.stop()

    st.subheader("📊 F1-Score Comparison")
    fm = results.get("feature_mode")
    if fm:
        st.caption(f"Feature mode: **{fm}**" + (f" · text column `{results.get('text_col')}`" if results.get("text_col") else ""))
    baseline_f1 = (results.get("baseline_real_only") or {}).get("test_f1")
    naive_f1 = (results.get("naive_augmentation") or {}).get("test_f1")
    bandit_f1 = (results.get("bandit_guided") or {}).get("test_f1")

    m1, m2, m3 = st.columns(3)
    m1.markdown(_f1_card("Baseline (real data only)", baseline_f1), unsafe_allow_html=True)
    m2.markdown(_f1_card("Naive augmentation", naive_f1, baseline_f1), unsafe_allow_html=True)
    m3.markdown(_f1_card("Bandit-guided (ours)", bandit_f1, baseline_f1), unsafe_allow_html=True)

    if all(v is not None for v in [baseline_f1, naive_f1, bandit_f1]):
        chart_df = pd.DataFrame({
            "Method": ["Baseline", "Naive Aug.", "Bandit (ours)"],
            "Test F1": [baseline_f1, naive_f1, bandit_f1],
        })
        st.bar_chart(chart_df.set_index("Method"), height=280, use_container_width=True)

    st.divider()
    st.subheader("🎰 Bandit Statistics")
    bs = (results.get("bandit_guided") or {}).get("bandit_summary") or {}
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total steps", bs.get("total_steps", "—"))
    b2.metric("Accepts", bs.get("total_accepts", "—"))
    b3.metric("Rejects", bs.get("total_rejects", "—"))
    rate = bs.get("acceptance_rate")
    b4.metric("Acceptance rate", f"{rate:.1%}" if rate is not None else "—")
    with st.expander("Full bandit summary"):
        st.json(bs)

    st.divider()
    st.subheader("📥 Download Outputs")
    out_dir = Path(cfg.OUTPUT_DIR)
    files_meta = {
        "curated.csv": ("text/csv", "Bandit-selected synthetic samples"),
        "augmented.csv": ("text/csv", "Original + curated merged dataset"),
        "report.json": ("application/json", "Full evaluation report"),
    }
    dl_cols = st.columns(3)
    for (fname, (mime, desc)), col in zip(files_meta.items(), dl_cols):
        fpath = out_dir / fname
        with col:
            if fpath.exists():
                col.download_button(f"⬇ {fname}", data=fpath.read_bytes(),
                                    file_name=fname, mime=mime,
                                    help=desc, use_container_width=True)
            else:
                col.button(f"⬇ {fname}", disabled=True, use_container_width=True)

    st.divider()
    with st.expander("📋 Full results JSON"):
        st.json(results)
