// components/CriticSection.jsx — Phase 4 Critic Verification Review UI
import { useState } from 'react'
import './CriticSection.css'

export default function CriticSection({ criticResponse }) {
  const [filter, setFilter] = useState('all') // 'all', 'ACCEPTED', 'QUESTIONABLE', 'REJECTED', 'REQUIRES_REINVESTIGATION'

  if (!criticResponse || !criticResponse.reviews) {
    return null
  }

  const { summary, reviews } = criticResponse

  const filteredReviews = reviews.filter((r) => {
    if (filter === 'ACCEPTED') return r.status === 'ACCEPTED'
    if (filter === 'QUESTIONABLE') return r.status === 'QUESTIONABLE'
    if (filter === 'REJECTED') return r.status === 'REJECTED'
    if (filter === 'REQUIRES_REINVESTIGATION') return r.status === 'REQUIRES_REINVESTIGATION' || r.requires_reinvestigation
    return true
  })

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'ACCEPTED': return 'status-accepted'
      case 'QUESTIONABLE': return 'status-questionable'
      case 'REJECTED': return 'status-rejected'
      case 'REQUIRES_REINVESTIGATION': return 'status-reinvestigate'
      default: return 'status-questionable'
    }
  }

  const getCardStatusClass = (status) => {
    switch (status) {
      case 'ACCEPTED': return 'card-status-accepted'
      case 'QUESTIONABLE': return 'card-status-questionable'
      case 'REJECTED': return 'card-status-rejected'
      case 'REQUIRES_REINVESTIGATION': return 'card-status-reinvestigate'
      default: return ''
    }
  }

  const formatStatusLabel = (status) => {
    switch (status) {
      case 'ACCEPTED': return '✓ Accepted'
      case 'QUESTIONABLE': return '⚠ Questionable'
      case 'REJECTED': return '✕ Rejected'
      case 'REQUIRES_REINVESTIGATION': return '🔄 Re-investigate'
      default: return status
    }
  }

  return (
    <div className="critic-section-container fade-in">
      {/* ── Summary & Verdict Overview ────────────────────────────── */}
      <div className="critic-overview-card">
        <div className="critic-overview-header">
          <div className="critic-title-row">
            <span>🛡 Critic Agent Verification Review</span>
            <span className="badge badge-cyan">Phase 4 Active</span>
          </div>

          {/* Distribution stat chips */}
          <div className="critic-stat-chips">
            <div className="critic-stat-chip chip-accepted">
              <span>✓</span>
              <span>{summary.accepted} Accepted</span>
            </div>
            {summary.questionable > 0 && (
              <div className="critic-stat-chip chip-questionable">
                <span>⚠</span>
                <span>{summary.questionable} Questionable</span>
              </div>
            )}
            {summary.rejected > 0 && (
              <div className="critic-stat-chip chip-rejected">
                <span>✕</span>
                <span>{summary.rejected} Rejected</span>
              </div>
            )}
            {summary.requires_reinvestigation > 0 && (
              <div className="critic-stat-chip chip-reinvestigate">
                <span>🔄</span>
                <span>{summary.requires_reinvestigation} Re-investigate</span>
              </div>
            )}
          </div>
        </div>

        {/* High-level synthesis headline */}
        <div className="critic-status-headline">
          <strong>Overall Verdict:</strong> {summary.overall_status}
        </div>

        {/* Critical issues banner if any */}
        {summary.critical_issues && summary.critical_issues.length > 0 && (
          <div className="planner-warnings" style={{ margin: 0 }}>
            <div className="warnings-header">
              <span>⚠</span>
              <span>Methodological & Data Quality Warnings ({summary.critical_issues.length})</span>
            </div>
            {summary.critical_issues.map((issue, idx) => (
              <div key={idx} className="warning-item">• {issue}</div>
            ))}
          </div>
        )}

        {/* Filters */}
        <div className="analysis-filters" style={{ marginTop: '4px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Filter Findings:</span>
          <button
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All Findings ({summary.total_results})
          </button>
          <button
            className={`filter-btn ${filter === 'ACCEPTED' ? 'active' : ''}`}
            onClick={() => setFilter('ACCEPTED')}
          >
            Accepted ({summary.accepted})
          </button>
          {summary.questionable > 0 && (
            <button
              className={`filter-btn ${filter === 'QUESTIONABLE' ? 'active' : ''}`}
              onClick={() => setFilter('QUESTIONABLE')}
            >
              Questionable ({summary.questionable})
            </button>
          )}
          {summary.rejected > 0 && (
            <button
              className={`filter-btn ${filter === 'REJECTED' ? 'active' : ''}`}
              onClick={() => setFilter('REJECTED')}
            >
              Rejected ({summary.rejected})
            </button>
          )}
          {summary.requires_reinvestigation > 0 && (
            <button
              className={`filter-btn ${filter === 'REQUIRES_REINVESTIGATION' ? 'active' : ''}`}
              onClick={() => setFilter('REQUIRES_REINVESTIGATION')}
            >
              Re-investigate ({summary.requires_reinvestigation})
            </button>
          )}
        </div>
      </div>

      {/* ── Critic Review Cards Grid ─────────────────────────────── */}
      <div className="critic-cards-grid">
        {filteredReviews.map((r) => (
          <div
            key={r.task_id}
            className={`critic-card ${getCardStatusClass(r.status)}`}
          >
            {/* Header */}
            <div className="critic-card-header">
              <div className="critic-meta-wrap">
                <span className="critic-task-id">{r.task_id}</span>
                <span className="critic-method-name">{r.method}</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className={`badge-sig ${r.confidence_level === 'HIGH' ? 'sig-yes' : r.confidence_level === 'MEDIUM' ? 'sig-no' : 'sig-no'}`}>
                  {r.confidence_level} Confidence ({(r.confidence * 100).toFixed(0)}%)
                </span>
                <span className={`verdict-badge ${getStatusBadgeClass(r.status)}`}>
                  {formatStatusLabel(r.status)}
                </span>
              </div>
            </div>

            {/* Evaluated variables */}
            {r.variables && r.variables.length > 0 && (
              <div className="result-target-row">
                <span>Evaluated:</span>
                <span>{r.variables.join(' ↔ ')}</span>
              </div>
            )}

            {/* Scientific Assessment Summary */}
            <div className="critic-summary-box">
              <strong>Assessment: </strong>{r.summary}
            </div>

            {/* Verified Empirical Evidence */}
            {r.evidence && r.evidence.length > 0 && (
              <div className="critic-section-sub">
                <span className="critic-sub-title">Verified Empirical Evidence</span>
                <div className="critic-evidence-list">
                  {r.evidence.map((ev, idx) => (
                    <div key={idx} className="critic-evidence-item">
                      <span style={{ color: 'var(--accent-cyan)' }}>•</span>
                      <span>{ev}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Validity & Method */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.78rem' }}>
              <div>
                <span className="critic-sub-title">Method Appropriateness</span>
                <div style={{ color: 'var(--text-secondary)', marginTop: '2px', lineHeight: '1.4' }}>
                  {r.method_assessment}
                </div>
              </div>
              <div>
                <span className="critic-sub-title">Statistical Validity</span>
                <div style={{ color: 'var(--text-secondary)', marginTop: '2px', lineHeight: '1.4' }}>
                  {r.statistical_validity}
                </div>
              </div>
            </div>

            {/* Data Quality & Assumption Concerns */}
            {(r.data_quality_concerns.length > 0 || r.assumption_concerns.length > 0) && (
              <div className="critic-section-sub" style={{ background: 'rgba(245, 158, 11, 0.05)', padding: '8px 10px', borderRadius: '4px', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                <span className="critic-sub-title" style={{ color: '#fbbf24' }}>Caveats & Assumptions</span>
                <div className="critic-evidence-list">
                  {r.data_quality_concerns.map((c, idx) => (
                    <div key={`dq-${idx}`} className="critic-evidence-item" style={{ color: '#fbbf24' }}>
                      <span>⚠</span>
                      <span>{c}</span>
                    </div>
                  ))}
                  {r.assumption_concerns.map((c, idx) => (
                    <div key={`as-${idx}`} className="critic-evidence-item" style={{ color: '#fbbf24' }}>
                      <span>⚠</span>
                      <span>{c}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reinvestigation Proposal Callout */}
            {r.requires_reinvestigation && r.recommended_method && (
              <div className="reinvestigate-callout">
                <div className="reinvestigate-header">
                  <span>🔄</span>
                  <span>Recommended Re-investigation Action</span>
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', lineHeight: '1.4' }}>
                  {r.recommendation}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Suggested Alternative:</span>
                  <span className="reinvestigate-method-tag">{r.recommended_method}</span>
                </div>
              </div>
            )}

            {/* Actionable Guidance if not reinvestigate */}
            {(!r.requires_reinvestigation || !r.recommended_method) && r.recommendation && (
              <div className="critic-section-sub">
                <span className="critic-sub-title">Guidance</span>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  {r.recommendation}
                </div>
              </div>
            )}

            {/* Reason codes */}
            {r.reason_codes && r.reason_codes.length > 0 && (
              <div className="reason-codes-wrap">
                {r.reason_codes.map((code, idx) => (
                  <span key={idx} className="reason-chip">{code}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
