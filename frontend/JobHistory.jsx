export default function JobHistory({ jobs, onSelectJob, loading, error, onRefresh }) {
  if (loading && !jobs.length && !error) {
    return (
      <div className="page empty-page">
        <div className="empty-state">
          <div className="history-spinner-wrap" aria-hidden>
            <div className="history-spinner" />
          </div>
          <p>Loading jobs from API…</p>
        </div>
      </div>
    );
  }

  if (error && !jobs.length) {
    return (
      <div className="page empty-page">
        <div className="empty-state">
          <div className="empty-icon">⚠</div>
          <p className="history-error-msg">{error}</p>
          <button type="button" className="history-refresh-btn" onClick={onRefresh}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!jobs.length) {
    return (
      <div className="page empty-page">
        <div className="empty-state">
          <div className="empty-icon">◷</div>
          <p>No jobs on the server yet. Submit a pipeline run from the sidebar.</p>
          <button type="button" className="history-refresh-btn" onClick={onRefresh}>
            Refresh
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header history-header-row">
        <div>
          <h2 className="page-title">Job History</h2>
          <p className="page-sub">
            {jobs.length} job{jobs.length !== 1 ? "s" : ""} on server
            {loading ? " · updating…" : ""}
          </p>
        </div>
        <button
          type="button"
          className="history-refresh-btn"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <p className="history-inline-warn" role="status">
          {error} — showing last loaded list.
        </p>
      )}

      <div className="history-list">
        {jobs.map(job => (
          <div
            key={job.job_id}
            className="history-row"
            onClick={() => onSelectJob(job)}
            onKeyDown={e => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelectJob(job);
              }
            }}
            role="button"
            tabIndex={0}
          >
            <div className={`history-status-dot ${job.status || "pending"}`} />
            <div className="history-main">
              <code className="history-id">{job.job_id}</code>
              <span className={`history-badge ${job.status || "pending"}`}>{job.status || "pending"}</span>
            </div>
            <div className="history-arrow">→</div>
          </div>
        ))}
      </div>
    </div>
  );
}
