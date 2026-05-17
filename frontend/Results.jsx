import { useState } from "react";
import { API_BASE } from "./apiConfig";

export default function Results({ results, currentJob }) {
  const [showFullReport, setShowFullReport] = useState(false);

  if (!results) return null;

  const baseline = results.baseline_test_f1 ?? null;
  const naive    = results.naive_test_f1    ?? null;
  const bestVal  = results.best_val_f1      ?? null;
  const bandit   = bestVal ?? results.bandit_test_f1 ?? null;
  const bs       = results.bandit_summary   ?? {};

  const methods = [
    { key: "baseline", label: "Baseline", sub: "Real data only", f1: baseline, color: "#4a5568" },
    { key: "naive",    label: "Naive Aug.", sub: "All synthetic added", f1: naive, color: "#667eea", delta: baseline },
    { key: "bandit",   label: "Curata", sub: bestVal ? "Best checkpoint F1" : "Bandit-guided selection", f1: bandit, color: "#48bb78", delta: baseline, highlight: true },
  ];

  const maxF1 = Math.max(...methods.map(m => m.f1 ?? 0));

  const download = (filename) => {
    window.open(`${API_BASE}/jobs/${currentJob?.job_id}/download/${filename}`, "_blank");
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h2 className="page-title">Results</h2>
          <p className="page-sub">Dataset: <strong>{results.dataset || "—"}</strong></p>
          <p className="page-sub">Classifier: <strong>{currentJob?.classifier_model || results.classifier_model || "—"}</strong></p>
        </div>
        <div className="stat-pills">
          <div className="stat-pill success">
            <span className="pill-val">Done</span>
            <span className="pill-lbl">Status</span>
          </div>
          {bandit && baseline && (
            <div className={`stat-pill ${bandit >= baseline ? "success" : "warn"}`}>
              <span className="pill-val">
                {bandit >= baseline ? "+" : ""}{((bandit - baseline) * 100).toFixed(1)}%
              </span>
              <span className="pill-lbl">vs Baseline</span>
            </div>
          )}
        </div>
      </div>

      {/* F1 Comparison */}
      <section className="results-section">
        <h3 className="section-title">F1-Score Comparison</h3>
        <div className="f1-cards">
          {methods.map(m => (
            <div key={m.key} className={`f1-card ${m.highlight ? "highlight" : ""}`}>
              {m.highlight && <div className="highlight-tag">OUR METHOD</div>}
              <div className="f1-label">{m.label}</div>
              <div className="f1-sub">{m.sub}</div>
              <div className="f1-score" style={{ color: m.highlight ? "#48bb78" : undefined }}>
                {m.f1 != null ? m.f1.toFixed(4) : "—"}
              </div>
              {m.delta != null && m.f1 != null && (
                <div className={`f1-delta ${m.f1 >= m.delta ? "pos" : "neg"}`}>
                  {m.f1 >= m.delta ? "▲" : "▼"} {Math.abs((m.f1 - m.delta) * 100).toFixed(2)}pp vs baseline
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Bar chart */}
        <div className="bar-chart">
          {methods.filter(m => m.f1 != null).map(m => {
            const heightPct = maxF1 > 0 ? (m.f1 / maxF1) * 100 : 0;
            return (
              <div key={m.key} className="bar-col">
                <span className="bar-val">{m.f1.toFixed(3)}</span>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${m.highlight ? "accent" : ""}`}
                    style={{ height: `${heightPct}%`, background: m.color }}
                  />
                </div>
                <span className="bar-lbl">{m.label}</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* Bandit Stats */}
      <section className="results-section">
        <h3 className="section-title">Thompson Sampling Bandit Stats</h3>
        <div className="bandit-grid">
          {[
            { label: "Total Steps", val: bs.total_steps },
            { label: "Accepted", val: bs.total_accepts },
            { label: "Rejected", val: bs.total_rejects },
            {
              label: "Acceptance Rate",
              val: bs.acceptance_rate != null ? `${(bs.acceptance_rate * 100).toFixed(1)}%` : null,
              accent: true,
            },
            { label: "Curated Size", val: results.curated_size },
          ].map(item => (
            <div key={item.label} className={`bandit-card ${item.accent ? "accent-card" : ""}`}>
              <div className="bandit-label">{item.label}</div>
              <div className="bandit-val">{item.val ?? "—"}</div>
            </div>
          ))}
        </div>

        {bs.acceptance_rate != null && (
          <div className="acceptance-bar-wrap">
            <span className="acceptance-label">Accept / Reject ratio</span>
            <div className="acceptance-track">
              <div className="acceptance-fill accept" style={{ width: `${bs.acceptance_rate * 100}%` }} />
              <div className="acceptance-fill reject" style={{ width: `${(1 - bs.acceptance_rate) * 100}%` }} />
            </div>
            <div className="acceptance-legend">
              <span><span className="dot accept-dot" /> Accept {(bs.acceptance_rate * 100).toFixed(1)}%</span>
              <span><span className="dot reject-dot" /> Reject {((1 - bs.acceptance_rate) * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </section>

      {/* Downloads */}
      {currentJob && (
        <section className="results-section">
          <h3 className="section-title">Download Outputs</h3>
          <div className="download-cards">
            {[
              { file: "curated.csv",   icon: "📊", title: "Curated Data",    desc: "Bandit-selected synthetic samples" },
              { file: "augmented.csv", icon: "🔗", title: "Augmented Data",  desc: "Original + curated merged dataset" },
              { file: "report.json",   icon: "📋", title: "Full Report",     desc: "Complete evaluation metrics JSON" },
            ].map(d => (
              <button key={d.file} className="download-card" onClick={() => download(d.file)}>
                <span className="dl-icon">{d.icon}</span>
                <div className="dl-info">
                  <span className="dl-title">{d.title}</span>
                  <span className="dl-desc">{d.desc}</span>
                </div>
                <DownloadIcon />
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Full JSON report */}
      <section className="results-section">
        <button className="toggle-report" onClick={() => setShowFullReport(v => !v)}>
          {showFullReport ? "▼" : "▶"} Full Report JSON
        </button>
        {showFullReport && (
          <pre className="json-viewer">{JSON.stringify(results.full_report ?? results, null, 2)}</pre>
        )}
      </section>
    </div>
  );
}

const DownloadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
  </svg>
);