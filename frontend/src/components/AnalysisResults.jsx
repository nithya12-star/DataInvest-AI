// components/AnalysisResults.jsx — Phase 3 Statistical Results Display
import { useState } from 'react'
import VegaChart from './VegaChart'
import './AnalysisResults.css'

function fmtNum(val, decimals = 4) {
  if (val === null || val === undefined) return '—'
  if (typeof val === 'number') {
    if (Math.abs(val) < 0.0001 && val !== 0) {
      return val.toExponential(4)
    }
    return val.toFixed(decimals)
  }
  return String(val)
}

export default function AnalysisResults({ analysisResponse }) {
  const [filter, setFilter] = useState('all') // 'all', 'success', 'failed'

  if (!analysisResponse || !analysisResponse.results) {
    return null
  }

  const { total_tasks, successful_tasks, failed_tasks, results } = analysisResponse

  const filteredResults = results.filter((r) => {
    if (filter === 'success') return r.status === 'success'
    if (filter === 'failed') return r.status === 'failed'
    return true
  })

  // 1. Correlation Renderer
  const renderCorrelation = (r) => {
    const res = r.results || {}
    const isSpearman = r.method?.toLowerCase().includes('spearman')
    const statSymbol = isSpearman ? 'ρ (rho)' : 'r (Pearson)'

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">Correlation Coefficient ({statSymbol})</span>
            <span className="stat-item-val highlight">{fmtNum(res.correlation_coefficient, 4)}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
              {res.association_strength} ({res.direction})
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">P-Value</span>
            <span className={`stat-item-val ${res.p_value_below_0_05 ? 'significant' : ''}`}>
              {res.p_value_formatted || fmtNum(res.p_value, 4)}
            </span>
            <span className={`badge-sig ${res.p_value_below_0_05 ? 'sig-yes' : 'sig-no'}`}>
              {res.p_value_below_0_05 ? 'p < 0.05 (Statistically Significant)' : 'p ≥ 0.05 (Not Significant)'}
            </span>
          </div>
        </div>

        <table className="results-table">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Sample Size (N)</th>
              <th>Mean</th>
              <th>Std Dev</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{res.column_a}</td>
              <td>{res.sample_size}</td>
              <td>{fmtNum(res.mean_a, 2)}</td>
              <td>{fmtNum(res.std_a, 2)}</td>
            </tr>
            <tr>
              <td>{res.column_b}</td>
              <td>{res.sample_size}</td>
              <td>{fmtNum(res.mean_b, 2)}</td>
              <td>{fmtNum(res.std_b, 2)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    )
  }

  // 2. Two-Group Comparison Renderer
  const renderTwoGroup = (r) => {
    const res = r.results || {}
    const g1 = res.group_1 || {}
    const g2 = res.group_2 || {}

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">Test Statistic (t / U)</span>
            <span className="stat-item-val highlight">{fmtNum(res.statistic, 4)}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Mean Difference: {fmtNum(res.mean_difference, 2)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">P-Value</span>
            <span className={`stat-item-val ${res.p_value_below_0_05 ? 'significant' : ''}`}>
              {res.p_value_formatted || fmtNum(res.p_value, 4)}
            </span>
            <span className={`badge-sig ${res.p_value_below_0_05 ? 'sig-yes' : 'sig-no'}`}>
              {res.p_value_below_0_05 ? 'p < 0.05 (Significant Difference)' : 'p ≥ 0.05 (No Significant Difference)'}
            </span>
          </div>
        </div>

        <table className="results-table">
          <thead>
            <tr>
              <th>Group Level</th>
              <th>N</th>
              <th>Mean</th>
              <th>Median</th>
              <th>Std Dev</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>{g1.name}</strong></td>
              <td>{g1.n}</td>
              <td>{fmtNum(g1.mean, 2)}</td>
              <td>{fmtNum(g1.median, 2)}</td>
              <td>{fmtNum(g1.std, 2)}</td>
            </tr>
            <tr>
              <td><strong>{g2.name}</strong></td>
              <td>{g2.n}</td>
              <td>{fmtNum(g2.mean, 2)}</td>
              <td>{fmtNum(g2.median, 2)}</td>
              <td>{fmtNum(g2.std, 2)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    )
  }

  // 3. Multi-Group ANOVA / Kruskal-Wallis Renderer
  const renderMultiGroup = (r) => {
    const res = r.results || {}
    const statKey = res.F_statistic !== undefined ? 'F-Statistic' : 'H-Statistic'
    const statVal = res.F_statistic !== undefined ? res.F_statistic : res.H_statistic

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">{statKey}</span>
            <span className="stat-item-val highlight">{fmtNum(statVal, 4)}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Groups: {res.num_groups} (Total N: {res.sample_size})
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">P-Value</span>
            <span className={`stat-item-val ${res.p_value_below_0_05 ? 'significant' : ''}`}>
              {res.p_value_formatted || fmtNum(res.p_value, 4)}
            </span>
            <span className={`badge-sig ${res.p_value_below_0_05 ? 'sig-yes' : 'sig-no'}`}>
              {res.p_value_below_0_05 ? 'p < 0.05 (Groups Differ Significantly)' : 'p ≥ 0.05 (No Significant Variation)'}
            </span>
          </div>
        </div>

        {res.groups && res.groups.length > 0 && (
          <table className="results-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>N</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std Dev</th>
              </tr>
            </thead>
            <tbody>
              {res.groups.map((g, idx) => (
                <tr key={idx}>
                  <td><strong>{g.name}</strong></td>
                  <td>{g.n}</td>
                  <td>{fmtNum(g.mean, 2)}</td>
                  <td>{fmtNum(g.median, 2)}</td>
                  <td>{fmtNum(g.std, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    )
  }

  // 4. Categorical Chi-Square Renderer
  const renderCategorical = (r) => {
    const res = r.results || {}
    const cTable = res.contingency_table

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">Chi-Square (χ²) / Cramér's V</span>
            <span className="stat-item-val highlight">
              χ² = {fmtNum(res.chi2_statistic, 2)}, V = {fmtNum(res.cramer_v, 3)}
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
              {res.cramer_v_interpretation} (df = {res.degrees_of_freedom})
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">P-Value</span>
            <span className={`stat-item-val ${res.p_value_below_0_05 ? 'significant' : ''}`}>
              {res.p_value_formatted || fmtNum(res.p_value, 4)}
            </span>
            <span className={`badge-sig ${res.p_value_below_0_05 ? 'sig-yes' : 'sig-no'}`}>
              {res.p_value_below_0_05 ? 'p < 0.05 (Significant Association)' : 'p ≥ 0.05 (Independent)'}
            </span>
          </div>
        </div>

        {cTable && (
          <div style={{ marginTop: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
              Contingency Matrix ({res.table_dimensions}):
            </div>
            <table className="contingency-table">
              <thead>
                <tr>
                  <th></th>
                  {cTable.col_labels?.map((col, idx) => (
                    <th key={idx}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cTable.row_labels?.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <th>{row}</th>
                    {cTable.matrix?.[rIdx]?.map((val, cIdx) => (
                      <td key={cIdx}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  // 5. Outlier Sensitivity Renderer
  const renderSensitivity = (r) => {
    const res = r.results || {}
    const base = res.baseline_metrics || {}
    const filt = res.filtered_metrics || {}

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">Potential Outliers Flagged</span>
            <span className="stat-item-val highlight">
              {res.outlier_count} ({fmtNum(res.outlier_percentage, 1)}%)
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Fences: [{fmtNum(res.lower_fence, 1)}, {fmtNum(res.upper_fence, 1)}]
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">Mean Shift After Trimming</span>
            <span className={`stat-item-val ${res.substantial_influence ? 'warning' : 'highlight'}`}>
              {res.mean_shift > 0 ? '+' : ''}{fmtNum(res.mean_shift, 2)} ({fmtNum(res.mean_shift_pct, 1)}%)
            </span>
            <span className={`badge-sig ${res.substantial_influence ? 'sig-yes' : 'sig-no'}`}>
              {res.substantial_influence ? 'Substantial Outlier Influence' : 'Minimal Outlier Influence'}
            </span>
          </div>
        </div>

        <table className="results-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Full Sample (N={res.original_sample_size})</th>
              <th>Trimmed Sample (N={res.filtered_sample_size})</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Mean</td>
              <td>{fmtNum(base.mean, 2)}</td>
              <td><strong>{fmtNum(filt.mean, 2)}</strong></td>
            </tr>
            <tr>
              <td>Median</td>
              <td>{fmtNum(base.median, 2)}</td>
              <td>{fmtNum(filt.median, 2)}</td>
            </tr>
            <tr>
              <td>Std Dev</td>
              <td>{fmtNum(base.std, 2)}</td>
              <td>{fmtNum(filt.std, 2)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    )
  }

  // 6. Distribution Analysis Renderer
  const renderDistribution = (r) => {
    const res = r.results || {}
    const norm = res.normality_test || {}

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">Shape & Symmetry</span>
            <span className="stat-item-val highlight">{res.symmetry}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Skewness: {fmtNum(res.skewness, 2)} | Kurtosis: {fmtNum(res.kurtosis, 2)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">Normality Assessment ({norm.method})</span>
            <span className="stat-item-val">
              p = {fmtNum(norm.p_value, 4)}
            </span>
            <span className={`badge-sig ${norm.is_normally_distributed ? 'sig-yes' : 'sig-no'}`}>
              {norm.is_normally_distributed ? 'Normal Distribution' : 'Non-Normal Distribution'}
            </span>
          </div>
        </div>
      </div>
    )
  }

  // 7. Missingness Audit Renderer
  const renderMissingness = (r) => {
    const res = r.results || {}

    return (
      <div className="result-body">
        <div className="stat-hero-block">
          <div className="stat-item">
            <span className="stat-item-label">Missingness Overview</span>
            <span className="stat-item-val highlight">
              {res.total_missing_cells} cells ({fmtNum(res.overall_missing_percentage, 1)}%)
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Total rows: {res.total_rows}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">Complete Cases</span>
            <span className="stat-item-val significant">
              {res.complete_cases} ({fmtNum(res.complete_cases_percentage, 1)}%)
            </span>
            <span className="badge-sig sig-yes">Usable for multi-variable models</span>
          </div>
        </div>
      </div>
    )
  }

  // 8. Failure Box
  const renderFailure = (r) => {
    return (
      <div className="failure-box">
        <span className="failure-code">⚠ Boundary / Skipped: {r.error?.code || 'EXECUTION_SKIPPED'}</span>
        <span className="failure-msg">{r.error?.message || 'The statistical test could not be evaluated.'}</span>
      </div>
    )
  }

  // Dispatch body renderer
  const renderResultBody = (r) => {
    if (r.status === 'failed') {
      return renderFailure(r)
    }

    switch (r.analysis_type) {
      case 'numerical_vs_numerical':
        return renderCorrelation(r)
      case 'two_groups_vs_numerical':
        return renderTwoGroup(r)
      case 'multiple_groups_vs_numerical':
        return renderMultiGroup(r)
      case 'categorical_vs_categorical':
        return renderCategorical(r)
      case 'outlier_sensitivity':
        return renderSensitivity(r)
      case 'distribution_analysis':
      case 'descriptive_statistics':
        return renderDistribution(r)
      case 'missing_value_investigation':
        return renderMissingness(r)
      default:
        return (
          <pre style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            {JSON.stringify(r.results, null, 2)}
          </pre>
        )
    }
  }

  return (
    <div className="analysis-results-container fade-in">
      {/* Summary card */}
      <div className="analysis-summary-card">
        <div className="analysis-summary-top">
          <div className="analysis-summary-title">
            <span>🔬 Statistical Execution Engine Results</span>
            <span className="badge badge-cyan">Phase 3</span>
          </div>

          <div className="analysis-summary-stats">
            <span className="stat-chip success">✓ {successful_tasks} Succeeded</span>
            {failed_tasks > 0 && (
              <span className="stat-chip warning">⚠ {failed_tasks} Skipped/Boundary</span>
            )}
            <span className="stat-chip info">⚡ Python & SciPy Verified</span>
          </div>
        </div>

        <div className="analysis-filters">
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Filter:</span>
          <button
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All Tasks ({total_tasks})
          </button>
          <button
            className={`filter-btn ${filter === 'success' ? 'active' : ''}`}
            onClick={() => setFilter('success')}
          >
            Successful ({successful_tasks})
          </button>
          {failed_tasks > 0 && (
            <button
              className={`filter-btn ${filter === 'failed' ? 'active' : ''}`}
              onClick={() => setFilter('failed')}
            >
              Boundaries ({failed_tasks})
            </button>
          )}
        </div>
      </div>

      {/* Grid of task result cards */}
      <div className="results-grid">
        {filteredResults.map((r) => (
          <div
            key={r.task_id}
            className={`result-card ${r.status === 'failed' ? 'card-failed' : ''}`}
          >
            <div className="result-header">
              <div className="result-meta-left">
                <span className="result-id">{r.task_id}</span>
                <span className="result-type">{r.method}</span>
              </div>
              <span className={`badge-sig ${r.status === 'success' ? 'sig-yes' : 'sig-no'}`}>
                {r.status === 'success' ? 'Executed' : 'Boundary'}
              </span>
            </div>

            {r.columns && r.columns.length > 0 && (
              <div className="result-target-row">
                <span>Variables:</span>
                <span>{r.columns.join(' ↔ ')}</span>
              </div>
            )}

            {renderResultBody(r)}

            {r.visual_evidence && (
              <VegaChart visualEvidence={r.visual_evidence} />
            )}

            {r.warnings && r.warnings.length > 0 && (
              <div className="result-warnings-list">
                {r.warnings.map((w, idx) => (
                  <div key={idx}>ℹ {w}</div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
