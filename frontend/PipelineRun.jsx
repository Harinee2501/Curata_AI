import { useEffect, useState, useRef } from "react";
import { API_BASE } from "./apiConfig";

/** Stages reflect only what the REST API exposes (pending / running / terminal). */
const STAGES = [
  {
    key: "pending",
    label: "Queued",
    desc: "Job accepted; waiting for the pipeline worker to start.",
  },
  {
    key: "running",
    label: "Running",
    desc: "Pipeline executing — synthesis, bandit filtering, training, and evaluation.",
  },
  {
    key: "done",
    label: "Complete",
    desc: "Finished successfully.",
  },
];

function stageIndexFromStatus(status) {
  if (status === "pending") return 0;
  if (status === "running") return 1;
  if (status === "done") return 2;
  if (status === "error") return 1;
  return 0;
}

function progressPercent(status, isError) {
  if (isError) return 35;
  if (status === "done") return 100;
  if (status === "running") return 55;
  if (status === "pending") return 18;
  return 10;
}

/** No hooks here — avoids conditional hook order and bad TS/JSX parsing after `useEffect`. */
export default function RunPipeline({ job, onJobComplete }) {
  if (!job) {
    return (
      <div className="page empty-page">
        <div className="empty-state">
          <div className="empty-icon">▷</div>
          <p>Submit a job from the sidebar to begin.</p>
        </div>
      </div>
    );
  }
  return <RunPipelineLoaded job={job} onJobComplete={onJobComplete} />;
}

function RunPipelineLoaded({ job, onJobComplete }) {
  const [status, setStatus] = useState(job.status || "pending");
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [stageIdx, setStageIdx] = useState(() => stageIndexFromStatus(job.status || "pending"));
  const intervalRef = useRef(null);
  const timerRef = useRef(null);
  const startRef = useRef(Date.now());
  const onJobCompleteRef = useRef(onJobComplete);
  onJobCompleteRef.current = onJobComplete;

  useEffect(() => {
    startRef.current = Date.now();
    setStatus(job.status || "pending");
    setError("");
    setStageIdx(stageIndexFromStatus(job.status || "pending"));

    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/jobs/${job.job_id}`);
        if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
        const data = await res.json();
        setStatus(data.status);
        setStageIdx(stageIndexFromStatus(data.status));

        if (data.status === "done") {
          clearInterval(intervalRef.current);
          clearInterval(timerRef.current);
          const rRes = await fetch(`${API_BASE}/jobs/${job.job_id}/results`);
          if (!rRes.ok) throw new Error(`Results fetch failed: ${rRes.status}`);
          const results = await rRes.json();
          onJobCompleteRef.current(results);
        } else if (data.status === "error") {
          clearInterval(intervalRef.current);
          clearInterval(timerRef.current);
          setError(data.message || "Pipeline failed");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        clearInterval(intervalRef.current);
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 3000);

    return () => {
      clearInterval(intervalRef.current);
      clearInterval(timerRef.current);
    };
  }, [job.job_id, job.status]);

  const isDone = status === "done";
  const isError = status === "error";
  const pct = progressPercent(status, isError);

  const stagesForUi = isError
    ? [
        STAGES[0],
        {
          key: "error",
          label: "Failed",
          desc: error || "Pipeline reported an error.",
        },
      ]
    : STAGES;

  const displayStageIdx = isError ? (stageIdx >= 1 ? 1 : 0) : Math.min(stageIdx, stagesForUi.length - 1);

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h2 className="page-title">Pipeline Execution</h2>
          <p className="page-sub">
            Job <code className="job-id-inline">{job.job_id.slice(0, 8)}…</code>
          </p>
        </div>
        <div className="stat-pills">
          <div className={`stat-pill ${isDone ? "success" : isError ? "error" : "running"}`}>
            <span className="pill-val">{isDone ? "Done" : isError ? "Error" : "Running"}</span>
            <span className="pill-lbl">Status</span>
          </div>
          <div className="stat-pill">
            <span className="pill-val">{elapsed}s</span>
            <span className="pill-lbl">Elapsed</span>
          </div>
        </div>
      </div>

      {isError && (
        <div className="error-banner">
          <strong>Pipeline Error:</strong> {error}
        </div>
      )}

      <div className="run-layout">
        <div className="progress-card">
          <div className="progress-header">
            <span>{isDone ? "Complete" : isError ? "Failed" : "Running…"}</span>
            <span className="pct">{pct}%</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${isDone ? "done" : isError ? "err" : "running"}`}
              style={{ width: `${pct}%` }}
            />
          </div>

          <div className="stages">
            {stagesForUi.map((s, i) => {
              const state = i < displayStageIdx ? "past" : i === displayStageIdx ? "active" : "future";
              return (
                <div key={s.key} className={`stage-row ${state} ${isError && s.key === "error" ? "stage-error" : ""}`}>
                  <div className="stage-dot">
                    {state === "past" ? "✓" : state === "active" ? <PulseIcon /> : "·"}
                  </div>
                  <div className="stage-info">
                    <span className="stage-label">{s.label}</span>
                    {state === "active" && <span className="stage-desc">{s.desc}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="job-info-card">
          <h4 className="card-title">Job Details</h4>
          <div className="kv-list">
            <div className="kv-row">
              <span className="kv-key">Job ID</span>
              <code className="kv-val mono">{job.job_id}</code>
            </div>
            <div className="kv-row">
              <span className="kv-key">Status</span>
              <span className="kv-val">{status}</span>
            </div>
          </div>

          {isDone && (
            <div className="done-notice">
              <span className="done-check">✓</span>
              Pipeline complete — switching to Results…
            </div>
          )}

          <div className="pipeline-legend">
            <h4 className="card-title" style={{ marginTop: "1.5rem" }}>
              Architecture
            </h4>
            {[
              ["LLM", "Generates synthetic candidate pool"],
              ["Thompson Sampling", "Bayesian bandit filters by quality"],
              ["PPO Agent", "Deep RL orchestrates the loop"],
              ["Classifier", "Measures augmentation quality"],
            ].map(([name, desc]) => (
              <div key={name} className="legend-row">
                <span className="legend-name">{name}</span>
                <span className="legend-desc">{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function PulseIcon() {
  return <span className="pulse-dot" />;
}
