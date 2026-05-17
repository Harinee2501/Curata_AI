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
        <div className="landing-badge">AUTONOMOUS DATA CURATION</div>
        <h1 className="landing-title">
          Curata <span className="title-accent">AI</span>
        </h1>
        <p className="landing-subtitle">
          LLM-driven synthesis · Thompson Sampling filtration · PPO agent control
        </p>
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
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <p className="drop-text">{dragging ? "Drop it" : "Drop your labeled CSV"}</p>
        <p className="drop-hint">or click to browse · minimum 30 rows</p>
      </div>

      <div className="pipeline-steps">
        {[
          { num: "01", label: "Synthesize", desc: "LLM generates a pool of synthetic samples from your data distribution" },
          { num: "02", label: "Filter", desc: "Thompson Sampling bandit accepts/rejects candidates by quality signal" },
          { num: "03", label: "Orchestrate", desc: "PPO agent governs iterations — generate, filter, retrain, evaluate, stop" },
        ].map(s => (
          <div key={s.num} className="step-card">
            <span className="step-num">{s.num}</span>
            <span className="step-label">{s.label}</span>
            <span className="step-desc">{s.desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
