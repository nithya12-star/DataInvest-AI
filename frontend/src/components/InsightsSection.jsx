// components/InsightsSection.jsx — Phase 6 Verified Insight Generation UI
import { useState } from 'react'
import './InsightsSection.css'

export default function InsightsSection({ insightsResponse }) {
  const [filter, setFilter] = useState('all') // 'all', 'STRONG', 'MODERATE', 'WEAK', 'unresolved'
  const [expandedEvidence, setExpandedEvidence] = useState({}) // insight_id -> boolean

  if (!insightsResponse || !insightsResponse.summary) {
    return null
  }

  const {
    summary,
    key_insights = [],
    unresolved_findings = [],
  } = insightsResponse

  const toggleEvidence = (id) => {
    setExpandedEvidence((prev) => ({
      ...prev,
      [id]: !prev[id],
    }))
  }

  const filteredInsights = key_insights.filter((ins) => {
    if (filter === 'all') return true
    if (filter === 'STRONG') return ins.strength === 'STRONG'
    if (filter === 'MODERATE') return ins.strength === 'MODERATE'
    if (filter === 'WEAK') return ins.strength === 'WEAK'
    return true
  })

  const formatPValue = (p) => {
    if (p === null || p === undefined) return 'N/A'
    if (p < 0.001) return 'p < 0.001'
    return `p = ${p.toFixed(4)}`
  }

  const getStrengthBadgeClass = (strength) => {
    switch (strength) {
      case 'STRONG': return 'badge-strong'
      case 'MODERATE': return 'badge-moderate'
      case 'WEAK': return 'badge-weak'
      case 'NOT_ASSESSED': return 'badge-not_assessed'
      default: return 'badge-weak'
    }
  }

  const getCardStrengthClass = (strength) => {
    switch (strength) {
      case 'STRONG': return 'strength-strong'
      case 'MODERATE': return 'strength-moderate'
      case 'WEAK': return 'strength-weak'
      case 'NOT_ASSESSED': return 'strength-not_assessed'
      default: return ''
    }
  }

  const formatTypeLabel = (type) => {
    switch (type) {
      case 'ASSOCIATION': return 'Bivariate Association'
      case 'GROUP_DIFFERENCE': return 'Group Difference'
      case 'CATEGORICAL_ASSOCIATION': return 'Categorical Dependency'
      case 'SENSITIVITY': return 'Outlier Sensitivity'
      case 'DESCRIPTIVE': return 'Distributional'
      default: return type.replace(/_/g, ' ')
    }
  }

  return (
    <div className="insights-section-container fade-in">
      {/* ── Summary & Overview Card ────────────────────────────────────── */}
      <div className="insights-overview-card">
        <div className="insights-overview-header">
          <div className="insights-title-row">
            <span>✨ Verified Analytical Insights</span>
            <span className="badge badge-cyan">Phase 6 Active</span>
          </div>

          <span className="confidence-chip">
            Derived from {summary.total_verified_findings} Verified Findings
          </span>
        </div>

        {/* Dynamic Executive Headline */}
        <div className="insights-headline-banner">
          <strong>Key Takeaway:</strong>{' '}
          {summary.top_insight_title
            ? `Primary insight identified: ${summary.top_insight_title}`
            : 'All analytical hypotheses verified and translated into actionable insights.'}
        </div>

        {/* Dynamic Statistics Chips */}
        <div className="insights-stat-chips">
          <div className="insights-stat-chip chip-total">
            <span>💡</span>
            <span>{summary.total_insights} Total Insights</span>
          </div>
          {summary.strong_insights > 0 && (
            <div className="insights-stat-chip chip-strong">
              <span>⭐</span>
              <span>{summary.strong_insights} Strong Magnitude</span>
            </div>
          )}
          {summary.moderate_insights > 0 && (
            <div className="insights-stat-chip chip-moderate">
              <span>🔹</span>
              <span>{summary.moderate_insights} Moderate Magnitude</span>
            </div>
          )}
          {summary.weak_insights > 0 && (
            <div className="insights-stat-chip chip-weak">
              <span>🔸</span>
              <span>{summary.weak_insights} Weak / Negligible</span>
            </div>
          )}
          {summary.unresolved_count > 0 && (
            <div className="insights-stat-chip chip-unresolved">
              <span>⚠</span>
              <span>{summary.unresolved_count} Unresolved Excluded</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Filter Bar ─────────────────────────────────────────────────── */}
      <div className="insights-filter-bar">
        <button
          type="button"
          className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          <span>All Insights</span>
          <span className="filter-count-badge">{key_insights.length}</span>
        </button>

        {summary.strong_insights > 0 && (
          <button
            type="button"
            className={`filter-btn ${filter === 'STRONG' ? 'active' : ''}`}
            onClick={() => setFilter('STRONG')}
          >
            <span>⭐ Strong</span>
            <span className="filter-count-badge">{summary.strong_insights}</span>
          </button>
        )}

        {summary.moderate_insights > 0 && (
          <button
            type="button"
            className={`filter-btn filter-cyan ${filter === 'MODERATE' ? 'active' : ''}`}
            onClick={() => setFilter('MODERATE')}
          >
            <span>🔹 Moderate</span>
            <span className="filter-count-badge">{summary.moderate_insights}</span>
          </button>
        )}

        {summary.weak_insights > 0 && (
          <button
            type="button"
            className={`filter-btn filter-amber ${filter === 'WEAK' ? 'active' : ''}`}
            onClick={() => setFilter('WEAK')}
          >
            <span>🔸 Weak</span>
            <span className="filter-count-badge">{summary.weak_insights}</span>
          </button>
        )}

        {unresolved_findings.length > 0 && (
          <button
            type="button"
            className={`filter-btn filter-rose ${filter === 'unresolved' ? 'active' : ''}`}
            onClick={() => setFilter('unresolved')}
          >
            <span>⚠ Unresolved ({unresolved_findings.length})</span>
          </button>
        )}
      </div>

      {/* ── Content View: Insights List or Unresolved Findings ─────────── */}
      {filter === 'unresolved' ? (
        <div className="unresolved-cards-list fade-in">
          <div
            className="planner-warnings"
            style={{ borderColor: '#f59e0b', background: 'rgba(245, 158, 11, 0.08)' }}
          >
            <div className="warnings-header" style={{ color: '#fbbf24' }}>
              <span>🛡 Statistical Integrity Notice</span>
            </div>
            <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
              The following findings were flagged as Questionable or Rejected by the Critic Agent.
              They are segregated here for audit transparency and are NOT presented as verified
              insights.
            </div>
          </div>

          {unresolved_findings.map((uf, idx) => (
            <div
              key={idx}
              className={`unresolved-card ${
                uf.status === 'QUESTIONABLE' ? 'status-questionable' : 'status-rejected'
              }`}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                <span className="finding-id-badge">{uf.task_id}</span>
                <span
                  className={`status-badge ${
                    uf.status === 'QUESTIONABLE' ? 'status-questionable' : 'status-rejected'
                  }`}
                >
                  {uf.status === 'QUESTIONABLE' ? '⚠ Questionable' : '✕ Rejected'}
                </span>
              </div>
              <div style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                {uf.title}
              </div>
              <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                {uf.summary}
              </div>
              {uf.limitations?.length > 0 && (
                <div style={{ fontSize: '0.8rem', color: '#fca5a5' }}>
                  {uf.limitations.map((lim, lIdx) => (
                    <div key={lIdx}>• {lim}</div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="insights-cards-list fade-in">
          {filteredInsights.map((insight, idx) => {
            const isExpanded = !!expandedEvidence[insight.insight_id]
            const ev = insight.evidence || {}

            return (
              <div
                key={insight.insight_id}
                className={`insight-card ${getCardStrengthClass(insight.strength)}`}
              >
                {/* Header */}
                <div className="insight-card-header">
                  <div className="insight-meta-row">
                    <span className="insight-rank-badge">#{idx + 1}</span>
                    <span className="finding-id-badge">{insight.source_task_id}</span>
                    <span className={`strength-badge ${getStrengthBadgeClass(insight.strength)}`}>
                      {insight.strength} MAGNITUDE
                    </span>
                    <span className="type-tag">{formatTypeLabel(insight.interpretation_type)}</span>
                  </div>
                  <span className="status-badge status-verified">
                    {insight.verification_status === 'VERIFIED_AFTER_REINVESTIGATION'
                      ? '🔄 Verified (Re-tested)'
                      : '✓ Verified'}
                  </span>
                </div>

                {/* Insight Title */}
                <div className="insight-title-text">{insight.title}</div>

                {/* Main Insight Interpretation Text */}
                <div className="insight-body-box">{insight.insight}</div>

                {/* Practical Implication */}
                {insight.practical_implication && (
                  <div className="practical-implication-box">
                    <strong>Practical Context:</strong> {insight.practical_implication}
                  </div>
                )}

                {/* Limitations */}
                {insight.limitations?.length > 0 && (
                  <div className="insight-limitations">
                    {insight.limitations.map((lim, lIdx) => (
                      <div key={lIdx} className="limitation-item">
                        <span>ℹ</span>
                        <span>{lim}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Toggle Evidence Drawer Button */}
                <button
                  type="button"
                  className="evidence-toggle-btn"
                  onClick={() => toggleEvidence(insight.insight_id)}
                >
                  <span>🔬</span>
                  <span>{isExpanded ? 'Hide Evidence' : 'View Evidence'}</span>
                  <span>{isExpanded ? '▲' : '▼'}</span>
                </button>

                {/* Expandable Evidence Drawer */}
                {isExpanded && (
                  <div className="evidence-drawer fade-in">
                    <div className="evidence-drawer-title">Empirical Statistical Evidence</div>

                    <div className="evidence-grid">
                      <div className="evidence-metric-cell">
                        <span className="evidence-metric-label">Method</span>
                        <span className="evidence-metric-val">{ev.method || 'N/A'}</span>
                      </div>

                      <div className="evidence-metric-cell">
                        <span className="evidence-metric-label">
                          {ev.statistic_name || 'Statistic'}
                        </span>
                        <span className="evidence-metric-val">
                          {ev.statistic !== null && ev.statistic !== undefined
                            ? ev.statistic.toFixed(3)
                            : 'N/A'}
                        </span>
                      </div>

                      <div className="evidence-metric-cell">
                        <span className="evidence-metric-label">p-value</span>
                        <span className="evidence-metric-val">{formatPValue(ev.p_value)}</span>
                      </div>

                      <div className="evidence-metric-cell">
                        <span className="evidence-metric-label">Sample Size (n)</span>
                        <span className="evidence-metric-val">{ev.sample_size || 'N/A'}</span>
                      </div>

                      {ev.effect_size !== null && ev.effect_size !== undefined && (
                        <div className="evidence-metric-cell">
                          <span className="evidence-metric-label">Effect Size</span>
                          <span className="evidence-metric-val">
                            {ev.effect_size.toFixed(3)}{' '}
                            {ev.effect_size_label ? `(${ev.effect_size_label})` : ''}
                          </span>
                        </div>
                      )}
                    </div>

                    {ev.comparison_summary && (
                      <div
                        style={{
                          fontSize: '0.82rem',
                          color: '#e0f2fe',
                          padding: '8px 12px',
                          background: 'rgba(6, 182, 212, 0.1)',
                          borderRadius: '4px',
                          border: '1px solid rgba(6, 182, 212, 0.3)',
                        }}
                      >
                        <strong>Re-investigation Note:</strong> {ev.comparison_summary}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
