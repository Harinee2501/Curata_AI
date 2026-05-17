import { useState, useCallback, useEffect } from "react";
import Sidebar from "./Sidebar";
import DataPreview from "./DataPreview";
import RunPipeline from "./PipelineRun.jsx";
import Results from "./Results";
import JobHistory from "./JobHistory";
import { API_BASE } from "./apiConfig";

export default function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [uploadedFile, setUploadedFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState(null);
  const [augmentationMode, setAugmentationMode] = useState("balanced");
  const [classifierModel, setClassifierModel] = useState("logistic_regression");
  const [availableClassifiers, setAvailableClassifiers] = useState({});
  const [labelCol, setLabelCol] = useState("");
  const [textCol, setTextCol] = useState("");
  const [currentJob, setCurrentJob] = useState(null);
  const [jobResults, setJobResults] = useState(null);
  const [jobHistory, setJobHistory] = useState([]);
  const [jobHistoryLoading, setJobHistoryLoading] = useState(false);
  const [jobHistoryError, setJobHistoryError] = useState("");
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === "dark" ? "light" : "dark");

  const refreshJobList = useCallback(async () => {
    setJobHistoryError("");
    setJobHistoryLoading(true);
    try {
      const res = await fetch(`${API_BASE}/jobs`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setJobHistory(Array.isArray(data) ? data : []);
    } catch (e) {
      setJobHistoryError(e.message || "Could not load jobs");
    } finally {
      setJobHistoryLoading(false);
    }
  }, []);

  const fetchAvailableClassifiers = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/classifiers`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAvailableClassifiers(data);
    } catch (e) {
      console.error("Failed to fetch classifiers:", e);
    }
  }, []);

  useEffect(() => {
    refreshJobList();
    fetchAvailableClassifiers();
  }, [refreshJobList, fetchAvailableClassifiers]);

  useEffect(() => {
    if (activeTab === "history") refreshJobList();
  }, [activeTab, refreshJobList]);

  const handleFileUpload = (file, preview) => {
    setUploadedFile(file);
    setCsvPreview(preview);
    setJobResults(null);
    setCurrentJob(null);
    setTextCol("");
    setActiveTab("preview");
  };

  const handleJobSubmitted = useCallback(
    async (job) => {
      setCurrentJob(job);
      setActiveTab("run");
      await refreshJobList();
    },
    [refreshJobList]
  );

  const handleJobComplete = useCallback(
    async (results) => {
      setJobResults(results);
      await refreshJobList();
      setActiveTab("results");
    },
    [refreshJobList]
  );

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        uploadedFile={uploadedFile}
        onFileUpload={handleFileUpload}
        augmentationMode={augmentationMode}
        setAugmentationMode={setAugmentationMode}
        classifierModel={classifierModel}
        setClassifierModel={setClassifierModel}
        availableClassifiers={availableClassifiers}
        csvPreview={csvPreview}
        labelCol={labelCol}
        setLabelCol={setLabelCol}
        textCol={textCol}
        setTextCol={setTextCol}
        onJobSubmitted={handleJobSubmitted}
        hasResults={!!jobResults}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <main className="main-content">
        {activeTab === "upload" && (
          <UploadLanding onFileUpload={handleFileUpload} />
        )}
        {activeTab === "preview" && csvPreview && (
          <DataPreview csvPreview={csvPreview} labelCol={labelCol} />
        )}
        {activeTab === "run" && (
          <RunPipeline job={currentJob} onJobComplete={handleJobComplete} />
        )}
        {activeTab === "results" && jobResults && (
          <Results results={jobResults} currentJob={currentJob} />
        )}
        {activeTab === "history" && (
          <JobHistory
            jobs={jobHistory}
            loading={jobHistoryLoading}
            error={jobHistoryError}
            onRefresh={refreshJobList}
            onSelectJob={(job) => {
              setCurrentJob(job);
              setActiveTab("run");
            }}
          />
        )}
      </main>
    </div>
  );
}

function UploadLanding({ onFileUpload }) {
  const [dragging, setDragging] = useState(false);

  const processFile = (file) => {
    if (!file || !file.name.endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const raw = e.target?.result;
      if (typeof raw !== "string") return;
      const text = raw;
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

  return (
    <div className="upload-landing">
      <div className="landing-hero">
        <div className="landing-badge">
          <span className="badge-dot" />
          AUTONOMOUS DATA CURATION
        </div>
        <h1 className="landing-title">
          Curata <span className="title-accent">AI</span>
        </h1>
        <p className="landing-subtitle">
          LLM-driven synthesis · Thompson Sampling filtration · PPO agent control
        </p>
        <div className="landing-stats">
          <div className="stat-pill-hero">
            <span className="stat-value">3×</span>
            <span className="stat-label">Dataset growth</span>
          </div>
          <div className="stat-pill-hero">
            <span className="stat-value">94%</span>
            <span className="stat-label">Quality retention</span>
          </div>
          <div className="stat-pill-hero">
            <span className="stat-value">↑F1</span>
            <span className="stat-label">Classifier boost</span>
          </div>
        </div>
      </div>

      <div
        className={`drop-zone ${dragging ? "dragging" : ""}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => {
          e.preventDefault();
          setDragging(false);
          processFile(e.dataTransfer.files[0]);
        }}
        onClick={() => document.getElementById("csv-input").click()}
      >
        <input
          id="csv-input"
          type="file"
          accept=".csv"
          style={{ display: "none" }}
          onChange={e => processFile(e.target.files[0])}
        />
        <div className="drop-icon">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
          </svg>
        </div>
        <p className="drop-text">{dragging ? "Drop it" : "Drop your labeled CSV"}</p>
        <p className="drop-hint">or <span className="drop-link">click to browse</span> · minimum 30 rows</p>
      </div>

      <div className="pipeline-steps">
        {[
          { num: "01", label: "Synthesize", desc: "LLM generates a pool of synthetic samples from your data distribution" },
          { num: "02", label: "Filter", desc: "Thompson Sampling bandit accepts/rejects candidates by quality signal" },
          { num: "03", label: "Orchestrate", desc: "PPO agent governs iterations — generate, filter, retrain, evaluate, stop" },
        ].map(s => (
          <div key={s.num} className="step-card">
            <div className="step-num-wrap">
              <span className="step-num">{s.num}</span>
            </div>
            <span className="step-label">{s.label}</span>
            <span className="step-desc">{s.desc}</span>
          </div>
        ))}
      </div>

      <div className="landing-footer-pills">
        <span className="footer-pill">⊙ Production-grade pipeline</span>
        <span className="footer-pill">⊕ Data stays local</span>
        <span className="footer-pill">⚡ Async job execution</span>
      </div>
    </div>
  );
}
