export default function DataPreview({ csvPreview, labelCol }) {
    if (!csvPreview) return null;
    const { headers, rows, totalRows } = csvPreview;
  
    const labelDist = labelCol
      ? rows.reduce((acc, row) => {
          const val = row[labelCol] ?? "(null)";
          acc[val] = (acc[val] || 0) + 1;
          return acc;
        }, {})
      : {};
  
    const nullCount = rows.reduce((sum, row) =>
      sum + headers.filter(h => !row[h] || row[h] === "").length, 0);
  
    return (
      <div className="page">
        <div className="page-header">
          <div>
            <h2 className="page-title">Data Preview</h2>
            <p className="page-sub">First 30 rows · {totalRows.toLocaleString()} total</p>
          </div>
          <div className="stat-pills">
            <div className="stat-pill">
              <span className="pill-val">{totalRows.toLocaleString()}</span>
              <span className="pill-lbl">Rows</span>
            </div>
            <div className="stat-pill">
              <span className="pill-val">{headers.length}</span>
              <span className="pill-lbl">Columns</span>
            </div>
            {nullCount > 0 && (
              <div className="stat-pill warn">
                <span className="pill-val">{nullCount}</span>
                <span className="pill-lbl">Nulls</span>
              </div>
            )}
          </div>
        </div>
  
        <div className="preview-grid">
          <div className="table-card">
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    {headers.map(h => (
                      <th key={h} className={h === labelCol ? "label-col" : ""}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => (
                    <tr key={i}>
                      {headers.map(h => (
                        <td key={h} className={h === labelCol ? "label-col" : ""}>
                          {row[h] ?? <span className="null-val">null</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
  
          <div className="preview-sidebar">
            <div className="info-card">
              <h4 className="card-title">Schema</h4>
              <div className="schema-list">
                {headers.map(h => (
                  <div key={h} className={`schema-row ${h === labelCol ? "label-row" : ""}`}>
                    <span className="schema-col">{h}</span>
                    {h === labelCol && <span className="label-badge">LABEL</span>}
                  </div>
                ))}
              </div>
            </div>
  
            {labelCol && Object.keys(labelDist).length > 0 && (
              <div className="info-card">
                <h4 className="card-title">Class Distribution <span className="card-sub">(preview)</span></h4>
                <div className="dist-list">
                  {Object.entries(labelDist)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, count]) => {
                      const pct = Math.round((count / rows.length) * 100);
                      return (
                        <div key={cls} className="dist-row">
                          <span className="dist-cls">{cls}</span>
                          <div className="dist-bar-wrap">
                            <div className="dist-bar" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="dist-count">{count}</span>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
  
            {nullCount > 0 && (
              <div className="info-card warn-card">
                <span className="warn-icon">⚠</span>
                <p>{nullCount} null values detected — will be dropped during pipeline run.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }