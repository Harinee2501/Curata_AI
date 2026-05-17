import { useState } from "react";
import { API_BASE } from "./apiConfig";

export default function Sidebar({
  activeTab, setActiveTab,
  uploadedFile, onFileUpload,
  augmentationMode, setAugmentationMode,
  classifierModel, setClassifierModel,
  availableClassifiers,
  csvPreview, labelCol, setLabelCol,
  textCol, setTextCol,
  onJobSubmitted, hasResults,
  theme, onToggleTheme
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
      form.append("augmentation_mode", augmentationMode);
      form.append("classifier_model", classifierModel);
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
    { id: "upload",   icon: HomeIcon,    label: "Home" },
    { id: "preview",  icon: TableIcon,   label: "Data Preview",  disabled: !csvPreview },
    { id: "run",      icon: PlayIcon,    label: "Run Pipeline",  disabled: !uploadedFile },
    { id: "results",  icon: ChartIcon,   label: "Results",       disabled: !hasResults },
    { id: "history",  icon: HistoryIcon, label: "Job History" },
  ];

  return (
    <aside className="sidebar">
      {/* Logo + theme toggle */}
      <div className="sidebar-logo">
        <div className="logo-mark">C</div>
        <div className="logo-text-group">
          <span className="logo-text">Curata</span>
          <span className="logo-ai">AI</span>
        </div>
        <button
          className="theme-toggle"
          onClick={onToggleTheme}
          title="Toggle theme"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        <p className="nav-section-label">Navigation</p>
        {navItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${activeTab === item.id ? "active" : ""} ${item.disabled ? "disabled" : ""}`}
            onClick={() => !item.disabled && setActiveTab(item.id)}
            disabled={item.disabled}
          >
            <span className="nav-icon"><item.icon /></span>
            <span className="nav-label">{item.label}</span>
            {activeTab === item.id && <span className="nav-active-dot" />}
          </button>
        ))}
      </nav>

      <div className="sidebar-divider" />

      {/* Controls */}
      <div className="sidebar-controls">
        <p className="nav-section-label">Dataset</p>

        <div className="control-group">
          <div
            className="file-slot"
            onClick={() => document.getElementById("sidebar-csv").click()}
          >
            <input id="sidebar-csv" type="file" accept=".csv" style={{ display: "none" }} onChange={handleFileChange} />
            <span className="file-slot-icon">{uploadedFile ? <FileIcon /> : <UploadIcon />}</span>
            <div className="file-slot-text">
              {uploadedFile
                ? <>
                    <span className="file-name">{uploadedFile.name}</span>
                    {csvPreview && <span className="file-meta">{csvPreview.totalRows.toLocaleString()} rows · {csvPreview.headers.length} cols</span>}
                  </>
                : <>
                    <span className="file-name">Upload CSV file</span>
                    <span className="file-meta">Click to browse</span>
                  </>
              }
            </div>
          </div>
        </div>

        {csvPreview && (
          <div className="control-group">
            <label className="control-label">Label Column</label>
            <select className="control-select" value={labelCol} onChange={e => setLabelCol(e.target.value)}>
              <option value="">Select column…</option>
              {csvPreview.headers.map(h => <option key={h} value={h}>{h}</option>)}
            </select>
          </div>
        )}

        {csvPreview && labelCol && (
          <div className="control-group">
            <label className="control-label">Text column <span className="optional-tag">optional</span></label>
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
            Augmentation Mode
            <span className="control-value">{augmentationMode}</span>
          </label>
          <select
            className="control-select"
            value={augmentationMode}
            onChange={e => setAugmentationMode(e.target.value)}
          >
            <option value="fast">fast — lower compute, quicker results</option>
            <option value="balanced">balanced — default middle ground</option>
            <option value="thorough">thorough — highest quality, slower</option>
          </select>
          <div className="mode-note">
            Choose how aggressively the pipeline augments: pool size, RL steps, and threshold are set automatically.
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">
            Classifier Model
            <span className="control-value">{classifierModel}</span>
          </label>
          <select
            className="control-select"
            value={classifierModel}
            onChange={e => setClassifierModel(e.target.value)}
          >
            <option value="logistic_regression">Logistic Regression — Fast linear baseline</option>
            <option value="naive_bayes">Naive Bayes — Very fast, high-dimensional data</option>
            <option value="random_forest">Random Forest — Robust ensemble, handles non-linearity</option>
            <option value="xgboost">XGBoost — Gradient boosting, strong on tabular data</option>
            <option value="lightgbm">LightGBM — Gradient boosting, fast on large datasets</option>
            <option value="svm">SVM — RBF kernel, strong on small datasets</option>
          </select>
          <div className="mode-note">
            The model trained on your dataset. Affects pipeline quality and runtime.
          </div>
        </div>

        {error && <div className="sidebar-error">{error}</div>}

        <button
          className="run-button"
          onClick={handleRun}
          disabled={!uploadedFile || !labelCol || submitting}
        >
          {submitting
            ? <><SpinnerIcon /><span>Submitting…</span></>
            : <><PlayIcon /><span>Run Pipeline</span></>
          }
        </button>
      </div>

      {/* Footer with copyright */}
      <div className="sidebar-footer">
        <div className="sidebar-footer-row">
          <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="footer-link">
            <ApiIcon /> API Docs
          </a>
          <span className="footer-version">v2.0</span>
        </div>
        <div className="footer-copyright">
          © {new Date().getFullYear()} Curata AI. All rights reserved.
        </div>
      </div>
    </aside>
  );
}

// ── Icons ──────────────────────────────────────────────────────────────
const s = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
const HomeIcon    = () => <svg {...s}><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9,22 9,12 15,12 15,22"/></svg>;
const TableIcon   = () => <svg {...s}><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/></svg>;
const PlayIcon    = () => <svg {...s}><polygon points="5,3 19,12 5,21"/></svg>;
const ChartIcon   = () => <svg {...s}><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>;
const HistoryIcon = () => <svg {...s}><polyline points="1,4 1,10 7,10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>;
const FileIcon    = () => <svg {...s}><path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z"/><polyline points="13,2 13,9 20,9"/></svg>;
const UploadIcon  = () => <svg {...s}><polyline points="16,16 12,12 8,16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/></svg>;
const ApiIcon     = () => <svg {...s}><polyline points="16,18 22,12 16,6"/><polyline points="8,6 2,12 8,18"/></svg>;
const SpinnerIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>;
const SunIcon     = () => <svg {...s}><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>;
const MoonIcon    = () => <svg {...s}><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>;
