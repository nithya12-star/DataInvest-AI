// components/VerifiedFindingsSection.jsx — Phase 5 Verified Findings & Re-investigation UI
import { useState } from 'react'
import VegaChart from './VegaChart'
import './VerifiedFindingsSection.css'

export default function VerifiedFindingsSection({ verificationResponse }) {
  const [filter, setFilter] = useState('all') // 'all', 'VERIFIED', 'VERIFIED_AFTER_REINVESTIGATION', 'QUESTIONABLE', 'REJECTED', 'AUDIT'

  if (!verificationResponse || !verificationResponse.summary) {
    return null
  }

  const {
    summary,
    verified_findings = [],
    questionable_findings = [],
    rejected_findings = [],
    all_findings = [],
  } = verificationResponse

  const auditTrail =
    verificationResponse.reinvestigations ||
    verificationResponse.reinvestigation_audit_trail ||
    []

  // Combine all findings into a unified list
  const allFindings =
    all_findings.length > 0
      ? all_findings
      : [...verified_findings, ...questionable_findings, ...rejected_findings]


  // Filter based on active tab
  const filteredFindings = allFindings.filter((f) => {
    const s = f.status || f.final_status
    if (filter === 'all') return true
    if (filter === 'VERIFIED') return s === 'VERIFIED'
    if (filter === 'VERIFIED_AFTER_REINVESTIGATION') return s === 'VERIFIED_AFTER_REINVESTIGATION'
    if (filter === 'QUESTIONABLE') return s === 'QUESTIONABLE'
    if (filter === 'REJECTED') return s === 'REJECTED'
    return true
  })

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'VERIFIED': return 'status-verified'
      case 'VERIFIED_AFTER_REINVESTIGATION': return 'status-reinvestigated'
      case 'QUESTIONABLE': return 'status-questionable'
      case 'REJECTED': return 'status-rejected'
      default: return 'status-questionable'
    }
  }

  const getCardBorderClass = (status) => {
    switch (status) {
      case 'VERIFIED': return 'status-border-verified'
      case 'VERIFIED_AFTER_REINVESTIGATION': return 'status-border-reinvestigated'
      case 'QUESTIONABLE': return 'status-border-questionable'
      case 'REJECTED': return 'status-border-rejected'
      default: return ''
    }
  }

  const formatStatusLabel = (status) => {
    switch (status) {
      case 'VERIFIED': return '✓ Verified'
      case 'VERIFIED_AFTER_REINVESTIGATION': return '🔄 Verified (Re-investigated)'
      case 'QUESTIONABLE': return '⚠ Questionable'
      case 'REJECTED': return '✕ Rejected'
      default: return status || 'Unknown'
    }
  }

  const resolveConfidence = (finding) => {
    const val = finding.confidence !== undefined && finding.confidence !== null
      ? finding.confidence
      : finding.confidence_score
    if (typeof val === 'number' && !Number.isNaN(val)) {
      return val
    }
    return null
  }

  const resolvePValue = (finding) => {
    if (typeof finding.p_value === 'number' && !Number.isNaN(finding.p_value)) {
      return finding.p_value
    }
    const stat = finding.statistical_evidence
    if (stat) {
      if (typeof stat.p_value === 'number' && !Number.isNaN(stat.p_value)) {
        return stat.p_value
      }
      if (stat.normality_test && typeof stat.normality_test.p_value === 'number' && !Number.isNaN(stat.normality_test.p_value)) {
        return stat.normality_test.p_value
      }
    }
    return null
  }

  const formatPValue = (p) => {
    if (p === null || p === undefined || typeof p !== 'number' || Number.isNaN(p)) return 'N/A'
    if (p < 0.001) return 'p < 0.001'
    return `p = ${p.toFixed(4)}`
  }

  const resolveEffectSize = (finding) => {
    if (typeof finding.effect_size === 'number' && !Number.isNaN(finding.effect_size)) {
      return { val: finding.effect_size, label: finding.effect_size_label }
    }
    const stat = finding.statistical_evidence
    if (stat) {
      if (typeof stat.effect_size === 'number' && !Number.isNaN(stat.effect_size)) {
        return { val: stat.effect_size, label: stat.effect_size_type || stat.effect_size_label }
      }
    }
    return null
  }

  const formatEffectSize = (finding) => {
    const eff = resolveEffectSize(finding)
    if (!eff || eff.val === null || eff.val === undefined || typeof eff.val !== 'number' || Number.isNaN(eff.val)) return 'N/A'
    const label = eff.label ? ` (${eff.label})` : ''
    return `${eff.val.toFixed(3)}${label}`
  }

  return (
    <div className="verified-section-container fade-in">
      {/* ── Summary & Executive Verdict Banner ─────────────────────────── */}
      <div className="verified-overview-card">
        <div className="verified-overview-header">
          <div className="verified-title-row">
            <span>✨ Verified Findings & Synthesis</span>
            <span className="badge badge-cyan">Phase 5 Active</span>
          </div>

          <div
            className={`verified-integrity-badge integrity-${(
              summary.overall_integrity || 'moderate'
            ).toLowerCase()}`}
          >
            <span>🛡 Integrity:</span>
            <span>{summary.overall_integrity || 'MODERATE'}</span>
          </div>
        </div>

        {/* Executive synthesis verdict */}
        <div className="verified-verdict-headline">
          <strong>Executive Verdict:</strong>{' '}
          {summary.executive_verdict ||
            summary.overall_integrity ||
            'Empirical analysis verified across all dataset features.'}
        </div>

        {/* Statistics distribution chips */}
        <div className="verified-stat-chips">
          <div className="verified-stat-chip chip-verified">
            <span>✓</span>
            <span>{summary.verified} Directly Verified</span>
          </div>
          {summary.verified_after_reinvestigation > 0 && (
            <div className="verified-stat-chip chip-reinvestigated-verified">
              <span>🔄</span>
              <span>{summary.verified_after_reinvestigation} Verified After Re-investigation</span>
            </div>
          )}
          {summary.questionable > 0 && (
            <div className="verified-stat-chip chip-questionable">
              <span>⚠</span>
              <span>{summary.questionable} Questionable</span>
            </div>
          )}
          {summary.rejected > 0 && (
            <div className="verified-stat-chip chip-rejected">
              <span>✕</span>
              <span>{summary.rejected} Rejected</span>
            </div>
          )}
          {summary.total_reinvestigated > 0 && (
            <div className="verified-stat-chip chip-audit-count">
              <span>🔬</span>
              <span>{summary.total_reinvestigated} Re-investigations Run</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Filter Bar ─────────────────────────────────────────────────── */}
      <div className="verified-filter-bar">
        <button
          className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          <span>All Findings</span>
          <span className="filter-count-badge">{allFindings.length}</span>
        </button>

        <button
          className={`filter-btn ${filter === 'VERIFIED' ? 'active' : ''}`}
          onClick={() => setFilter('VERIFIED')}
        >
          <span>✓ Verified</span>
          <span className="filter-count-badge">{summary.verified}</span>
        </button>

        {summary.verified_after_reinvestigation > 0 && (
          <button
            className={`filter-btn filter-cyan ${
              filter === 'VERIFIED_AFTER_REINVESTIGATION' ? 'active' : ''
            }`}
            onClick={() => setFilter('VERIFIED_AFTER_REINVESTIGATION')}
          >
            <span>🔄 Re-investigated</span>
            <span className="filter-count-badge">
              {summary.verified_after_reinvestigation}
            </span>
          </button>
        )}

        {summary.questionable > 0 && (
          <button
            className={`filter-btn filter-amber ${filter === 'QUESTIONABLE' ? 'active' : ''}`}
            onClick={() => setFilter('QUESTIONABLE')}
          >
            <span>⚠ Questionable</span>
            <span className="filter-count-badge">{summary.questionable}</span>
          </button>
        )}

        {summary.rejected > 0 && (
          <button
            className={`filter-btn filter-rose ${filter === 'REJECTED' ? 'active' : ''}`}
            onClick={() => setFilter('REJECTED')}
          >
            <span>✕ Rejected</span>
            <span className="filter-count-badge">{summary.rejected}</span>
          </button>
        )}

        {auditTrail.length > 0 && (
          <button
            className={`filter-btn filter-indigo ${filter === 'AUDIT' ? 'active' : ''}`}
            onClick={() => setFilter('AUDIT')}
          >
            <span>🔬 Audit Trail</span>
            <span className="filter-count-badge">{auditTrail.length}</span>
          </button>
        )}
      </div>

      {/* ── Content View: Audit Trail or Findings Grid ─────────────────── */}
      {filter === 'AUDIT' ? (
        <div className="audit-trail-timeline fade-in">
          {auditTrail.map((record, idx) => (

            <div key={idx} className="audit-card">
              <div className="audit-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className="finding-id-badge">Audit #{idx + 1}</span>
                  <span className="badge badge-cyan">{record.original_task_id || record.source_task_id}</span>
                </div>
                <span className={`status-badge ${getStatusBadgeClass(record.final_status || record.status)}`}>
                  {formatStatusLabel(record.final_status || record.status)}
                </span>
              </div>

              {/* Execution Flow Diagram */}
              <div className="audit-steps-flow">
                <span className="audit-step">{record.original_method} (Initial)</span>
                <span className="audit-arrow">➔</span>
                <span className="audit-step" style={{ color: '#fbbf24' }}>
                  Critic Flag: {record.reinvestigation_reason || record.trigger_reason}
                </span>
                <span className="audit-arrow">➔</span>
                <span className="audit-step" style={{ color: '#22d3ee' }}>
                  {record.reinvestigated_method} (Re-test)
                </span>
                <span className="audit-arrow">➔</span>
                <span className="audit-step" style={{ color: '#34d399' }}>
                  {record.final_status || record.status}
                </span>
              </div>

              {/* Comparative Delta Box */}
              {(record.result_comparison || record.comparison) && (
                <div className="reinvestigation-delta-box">
                  <div className="delta-header">
                    <span>🔬 Comparative Statistical Delta</span>
                    <span>
                      {record.result_comparison?.status_shifted || record.comparison?.significance_consistent === false
                        ? '⚡ Status Shifted'
                        : '✓ Consistent'}
                    </span>
                  </div>
                  <div className="delta-comparison-grid">
                    <div>
                      <div className="delta-col-title">Original ({record.original_method})</div>
                      <div className="delta-item">{formatPValue(record.original_p_value)}</div>
                      <div className="delta-item">
                        Effect: {record.original_effect_size !== undefined && record.original_effect_size !== null
                          ? Number(record.original_effect_size).toFixed(3)
                          : (record.original_statistic !== undefined && record.original_statistic !== null
                            ? `Stat: ${Number(record.original_statistic).toFixed(3)}`
                            : 'N/A')}
                      </div>
                    </div>
                    <div>
                      <div className="delta-col-title">
                        Re-investigated ({record.reinvestigated_method})
                      </div>
                      <div className="delta-item">
                        {formatPValue(record.reinvestigated_p_value)}
                      </div>
                      <div className="delta-item">
                        Effect: {record.reinvestigated_effect_size !== undefined && record.reinvestigated_effect_size !== null
                          ? Number(record.reinvestigated_effect_size).toFixed(3)
                          : (record.reinvestigated_statistic !== undefined && record.reinvestigated_statistic !== null
                            ? `Stat: ${Number(record.reinvestigated_statistic).toFixed(3)}`
                            : 'N/A')}
                      </div>
                    </div>
                  </div>
                  {(record.result_comparison?.narrative_delta || record.comparison?.summary) && (
                    <div className="delta-narrative">
                      {record.result_comparison?.narrative_delta || record.comparison?.summary}
                    </div>
                  )}
                </div>
              )}

              {/* Visual Evidence: Original & Re-investigated Comparison */}
              {(record.original_visual_evidence || record.reinvestigated_visual_evidence) && (
                <div className="reinvestigation-charts-comparison">
                  {record.original_visual_evidence && (
                    <div className="reinvestigation-chart-col">
                      <div className="reinvestigation-chart-badge">
                        <span>Initial Method:</span> <strong>{record.original_method}</strong>
                      </div>
                      <VegaChart
                        visualEvidence={record.original_visual_evidence}
                        subtitle="Initial Analysis Chart"
                      />
                    </div>
                  )}
                  {record.reinvestigated_visual_evidence && (
                    <div className="reinvestigation-chart-col">
                      <div className="reinvestigation-chart-badge follow-up">
                        <span>Re-investigated:</span> <strong>{record.reinvestigated_method}</strong>
                      </div>
                      <VegaChart
                        visualEvidence={record.reinvestigated_visual_evidence}
                        subtitle="Follow-up Verification Chart"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="verified-cards-grid fade-in">
          {filteredFindings.map((finding) => {
            const finalStatus = finding.status || finding.final_status || 'QUESTIONABLE'
            const confidence = resolveConfidence(finding)
            const method = finding.method || finding.method_used || 'Statistical Analysis'
            const summaryText = finding.summary || finding.claim || 'Finding verified.'
            const targetVar = (Array.isArray(finding.variables) && finding.variables.length > 0)
              ? finding.variables.join(' ↔ ')
              : (finding.target_column || null)
            const tags = (Array.isArray(finding.tags) && finding.tags.length > 0)
              ? finding.tags
              : (Array.isArray(finding.reason_codes) ? finding.reason_codes : [])

            const isCriticObj = finding.critic_assessment && typeof finding.critic_assessment === 'object'
            const criticVerdict = isCriticObj
              ? (finding.critic_assessment.verdict || finalStatus)
              : (finding.critic_assessment ? finalStatus : null)
            const criticText = isCriticObj
              ? (finding.critic_assessment.summary || finding.critic_assessment.text || null)
              : (typeof finding.critic_assessment === 'string' ? finding.critic_assessment : null)
            const criticReasons = isCriticObj && Array.isArray(finding.critic_assessment.reasons) && finding.critic_assessment.reasons.length > 0
              ? finding.critic_assessment.reasons
              : (finding.verification_reason ? [finding.verification_reason] : [])

            return (
              <div
                key={finding.finding_id}
                className={`verified-card ${getCardBorderClass(finalStatus)}`}
              >
                {/* Header */}
                <div className="verified-card-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span className="finding-id-badge">{finding.finding_id}</span>
                    <span className={`status-badge ${getStatusBadgeClass(finalStatus)}`}>
                      {formatStatusLabel(finalStatus)}
                    </span>
                    <span className="confidence-chip">
                      Confidence: {confidence !== null ? `${(confidence * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="method-tag">
                    <span>🔬</span>
                    <span>{method}</span>
                  </div>
                </div>

                {/* Main Claim */}
                <div className="verified-claim-box">{summaryText}</div>

                {/* Key Metrics */}
                <div className="verified-metrics-row">
                  <div className="verified-metric-box">
                    <div className="verified-metric-label">Significance</div>
                    <div className="verified-metric-val">{formatPValue(resolvePValue(finding))}</div>
                  </div>

                  <div className="verified-metric-box">
                    <div className="verified-metric-label">Effect Size</div>
                    <div className="verified-metric-val">{formatEffectSize(finding)}</div>
                  </div>

                  {targetVar && (
                    <div className="verified-metric-box">
                      <div className="verified-metric-label">Target Variable</div>
                      <div className="verified-metric-val" style={{ color: 'var(--accent-cyan)' }}>
                        {targetVar}
                      </div>
                    </div>
                  )}

                  {finding.group_or_feature && (
                    <div className="verified-metric-box">
                      <div className="verified-metric-label">Feature / Group</div>
                      <div className="verified-metric-val">{finding.group_or_feature}</div>
                    </div>
                  )}
                </div>

                {/* Before vs After Re-investigation Comparison (if re-investigated) */}
                {finding.reinvestigation_record && (
                  <div className="reinvestigation-delta-box">
                    <div className="delta-header">
                      <span>🔄 Re-investigation Comparative Delta</span>
                      <span>
                        {finding.reinvestigation_record.result_comparison?.status_shifted ||
                         finding.reinvestigation_record.comparison?.significance_consistent === false
                          ? '⚡ Method Adjusted'
                          : '✓ Verified Robust'}
                      </span>
                    </div>

                    <div className="delta-comparison-grid">
                      <div>
                        <div className="delta-col-title">
                          Initial Test ({finding.reinvestigation_record.original_method})
                        </div>
                        <div className="delta-item">
                          {formatPValue(
                            finding.reinvestigation_record.original_p_value ??
                            finding.reinvestigation_record.comparison?.original_p_value ??
                            finding.reinvestigation_record.original_analysis?.results?.p_value
                          )}
                        </div>
                        <div className="delta-item">
                          Effect Size: {(() => {
                            const origEff = finding.reinvestigation_record.original_effect_size ??
                              finding.reinvestigation_record.comparison?.original_effect_size ??
                              finding.reinvestigation_record.original_analysis?.results?.effect_size
                            if (origEff !== undefined && origEff !== null) return Number(origEff).toFixed(3)
                            const origStat = finding.reinvestigation_record.original_statistic ??
                              finding.reinvestigation_record.comparison?.original_statistic ??
                              finding.reinvestigation_record.original_analysis?.results?.statistic ??
                              finding.reinvestigation_record.original_analysis?.results?.correlation_coefficient
                            if (origStat !== undefined && origStat !== null) return `Stat: ${Number(origStat).toFixed(3)}`
                            return 'N/A'
                          })()}
                        </div>
                      </div>
                      <div>
                        <div className="delta-col-title">
                          Alternative Test ({finding.reinvestigation_record.reinvestigated_method})
                        </div>
                        <div className="delta-item">
                          {formatPValue(
                            finding.reinvestigation_record.reinvestigated_p_value ??
                            finding.reinvestigation_record.comparison?.reinvestigated_p_value ??
                            finding.reinvestigation_record.reinvestigated_analysis?.results?.p_value
                          )}
                        </div>
                        <div className="delta-item">
                          Effect Size: {(() => {
                            const newEff = finding.reinvestigation_record.reinvestigated_effect_size ??
                              finding.reinvestigation_record.comparison?.reinvestigated_effect_size ??
                              finding.reinvestigation_record.reinvestigated_analysis?.results?.effect_size
                            if (newEff !== undefined && newEff !== null) return Number(newEff).toFixed(3)
                            const newStat = finding.reinvestigation_record.reinvestigated_statistic ??
                              finding.reinvestigation_record.comparison?.reinvestigated_statistic ??
                              finding.reinvestigation_record.reinvestigated_analysis?.results?.statistic ??
                              finding.reinvestigation_record.reinvestigated_analysis?.results?.correlation_coefficient
                            if (newStat !== undefined && newStat !== null) return `Stat: ${Number(newStat).toFixed(3)}`
                            return 'N/A'
                          })()}
                        </div>
                      </div>
                    </div>

                    {(finding.reinvestigation_record.result_comparison?.narrative_delta ||
                      finding.reinvestigation_record.comparison?.summary) && (
                      <div className="delta-narrative">
                        {finding.reinvestigation_record.result_comparison?.narrative_delta ||
                         finding.reinvestigation_record.comparison?.summary}
                      </div>
                    )}
                  </div>
                )}

                {/* Re-investigation Visual Evidence: Comparative Side-by-Side */}
                {finding.reinvestigation_record &&
                  (finding.reinvestigation_record.original_visual_evidence ||
                    finding.reinvestigation_record.reinvestigated_visual_evidence) && (
                    <div className="reinvestigation-charts-comparison">
                      {finding.reinvestigation_record.original_visual_evidence && (
                        <div className="reinvestigation-chart-col">
                          <div className="reinvestigation-chart-badge">
                            <span>Initial Method:</span>{' '}
                            <strong>{finding.reinvestigation_record.original_method}</strong>
                          </div>
                          <VegaChart
                            visualEvidence={finding.reinvestigation_record.original_visual_evidence}
                            subtitle="Initial Analysis Chart"
                          />
                        </div>
                      )}
                      {finding.reinvestigation_record.reinvestigated_visual_evidence && (
                        <div className="reinvestigation-chart-col">
                          <div className="reinvestigation-chart-badge follow-up">
                            <span>Re-investigated:</span>{' '}
                            <strong>{finding.reinvestigation_record.reinvestigated_method}</strong>
                          </div>
                          <VegaChart
                            visualEvidence={finding.reinvestigation_record.reinvestigated_visual_evidence}
                            subtitle="Follow-up Verification Chart"
                          />
                        </div>
                      )}
                    </div>
                  )}

                {/* Direct / Final Finding Visual Evidence */}
                {finding.visual_evidence &&
                  finding.visual_evidence.graph_id !== finding.reinvestigation_record?.original_visual_evidence?.graph_id &&
                  finding.visual_evidence.graph_id !== finding.reinvestigation_record?.reinvestigated_visual_evidence?.graph_id && (
                    <VegaChart visualEvidence={finding.visual_evidence} />
                )}

                {/* Critic Assessment Box */}
                {(finding.critic_assessment || finding.verification_reason) && (
                  <div className="critic-review-summary-box">
                    <div className="critic-review-header">
                      <span>🛡 Critic Agent Assessment</span>
                      {criticVerdict && <span>Verdict: {criticVerdict}</span>}
                    </div>
                    {criticText && (
                      <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: criticReasons.length > 0 ? '8px' : '0' }}>
                        {criticText}
                      </div>
                    )}
                    {criticReasons.length > 0 && (
                      <div className="critic-reasons-list">
                        {criticReasons.map((r, rIdx) => (
                          <div key={rIdx} className="critic-reason-item">
                            <span>•</span>
                            <span>{r}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Limitations */}
                {finding.limitations?.length > 0 && (
                  <div className="finding-limitations-box">
                    <div className="limitations-title">Methodological Limitations & Caveats</div>
                    {finding.limitations.map((lim, limIdx) => (
                      <div key={limIdx}>⚠ {lim}</div>
                    ))}
                  </div>
                )}

                {/* Tags */}
                {tags.length > 0 && (
                  <div className="finding-tags-row">
                    {tags.map((tag, tIdx) => (
                      <span key={tIdx} className="finding-tag">
                        #{tag}
                      </span>
                    ))}
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
