import { useState } from "react";
import { API_BASE } from "./apiConfig";

export default function Sidebar({
  activeTab, setActiveTab,
  uploadedFile, onFileUpload,
  targetSize, setTargetSize,
  csvPreview, labelCol, setLabelCol,
  textCol, setTextCol,
  onJobSubmitted, hasResults
}) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target.result;
      const lines = text.split("\n").filter(Boolean);
      const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
      const rows = lines.slice(1, 31).map(line => {
        const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g, ""));
        return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]));
      });
      onFileUpload(file, { headers, rows, totalRows: lines.length - 1, text });
    };
    reader.readAsText(file);
  };

  const handleRun = async () => {
    if (!uploadedFile || !labelCol) return;
    setSubmitting(true);
    setError("");
    try {
      const form = new FormData();
      form.append("csv_file", uploadedFile);
      form.append("label_col", labelCol);
      form.append("target_size", targetSize);
      if (textCol && textCol.trim()) {
        form.append("text_col", textCol.trim());
      }
      const res = await fetch(`${API_BASE}/jobs`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const job = await res.json();
      onJobSubmitted(job);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const navItems = [
    { id: "upload", icon: HomeIcon, label: "Home" },
    { id: "preview", icon: TableIcon, label: "Data Preview", disabled: !csvPreview },
    { id: "run", icon: PlayIcon, label: "Run Pipeline", disabled: !uploadedFile },
    { id: "results", icon: ChartIcon, label: "Results", disabled: !hasResults },
    { id: "history", icon: HistoryIcon, label: "Job History" },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-mark">C</span>
        <span className="logo-text">Curata</span>
        <span className="logo-ai">AI</span>
      </div>

      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${activeTab === item.id ? "active" : ""} ${item.disabled ? "disabled" : ""}`}
            onClick={() => !item.disabled && setActiveTab(item.id)}
            disabled={item.disabled}
          >
            <item.icon />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-divider" />

      <div className="sidebar-controls">
        <div className="control-group">
          <label className="control-label">Dataset</label>
          <div
            className="file-slot"
            onClick={() => document.getElementById("sidebar-csv").click()}
          >
            <input id="sidebar-csv" type="file" accept=".csv" style={{ display: "none" }} onChange={handleFileChange} />
            {uploadedFile
              ? <><FileIcon /><span className="file-name">{uploadedFile.name}</span></>
              : <><UploadIcon /><span className="file-placeholder">Upload CSV</span></>
            }
          </div>
          {csvPreview && (
            <span className="file-meta">{csvPreview.totalRows.toLocaleString()} rows · {csvPreview.headers.length} cols</span>
          )}
        </div>

        {csvPreview && (
          <div className="control-group">
            <label className="control-label">Label Column</label>
            <select
              className="control-select"
              value={labelCol}
              onChange={e => setLabelCol(e.target.value)}
            >
              <option value="">Select column…</option>
              {csvPreview.headers.map(h => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
        )}

        {csvPreview && labelCol && (
          <div className="control-group">
            <label className="control-label">Text column (optional)</label>
            <select
              className="control-select"
              value={textCol}
              onChange={e => setTextCol(e.target.value)}
              title="For IMDB-style CSVs: pick the review/text column, or leave Auto when there is only one non-numeric feature column."
            >
              <option value="">Auto-detect</option>
              {csvPreview.headers.filter(h => h !== labelCol).map(h => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
        )}

        <div className="control-group">
          <label className="control-label">
            Target Synthetic Rows
            <span className="control-value">{targetSize}</span>
          </label>
          <input
            type="range"
            min={10}
            max={500}
            step={10}
            value={targetSize}
            onChange={e => setTargetSize(Number(e.target.value))}
            className="control-range"
          />
          <div className="range-ticks">
            <span>10</span><span>250</span><span>500</span>
          </div>
        </div>

        {error && <div className="sidebar-error">{error}</div>}

        <button
          className="run-button"
          onClick={handleRun}
          disabled={!uploadedFile || !labelCol || submitting}
        >
          {submitting ? (
            <><SpinnerIcon /><span>Submitting…</span></>
          ) : (
            <><PlayIcon /><span>Run Pipeline</span></>
          )}
        </button>
      </div>

      <div className="sidebar-footer">
        <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="footer-link">
          <ApiIcon /> API Docs
        </a>
        <span className="footer-version">v2.0</span>
      </div>
    </aside>
  );
}

// ── Icons ──────────────────────────────────────────────────────────────
const s = { width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
const HomeIcon = () => <svg {...s}><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9,22 9,12 15,12 15,22"/></svg>;
const TableIcon = () => <svg {...s}><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/></svg>;
const PlayIcon = () => <svg {...s}><polygon points="5,3 19,12 5,21"/></svg>;
const ChartIcon = () => <svg {...s}><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>;
const HistoryIcon = () => <svg {...s}><polyline points="1,4 1,10 7,10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>;
const FileIcon = () => <svg {...s}><path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z"/><polyline points="13,2 13,9 20,9"/></svg>;
const UploadIcon = () => <svg {...s}><polyline points="16,16 12,12 8,16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/></svg>;
const ApiIcon = () => <svg {...s}><polyline points="16,18 22,12 16,6"/><polyline points="8,6 2,12 8,18"/></svg>;
const SpinnerIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 12a9 9 0 11-6.219-8.56" />
  </svg>
);