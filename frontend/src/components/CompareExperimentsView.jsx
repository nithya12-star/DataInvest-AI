// components/CompareExperimentsView.jsx — Compare Both View for Paired Experimentation
import { useState } from 'react'
import './CompareExperimentsView.css'
import BasePaperComparisonModal from './BasePaperComparisonModal'

function fmt(n, decimals = 1) {
  if (n === null || n === undefined) return '—'
  return typeof n === 'number' ? n.toFixed(decimals) : String(n)
}

function signedFmt(n, decimals = 1) {
  if (n === null || n === undefined) return '—'
  if (typeof n !== 'number') return String(n)
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(decimals)}`
}

function formatDimensionName(name) {
  if (!name) return ''
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function renderValueOrNotRecorded(val, suffix = '') {
  if (val === null || val === undefined) {
    return <span className="text-muted-italic">Not recorded</span>
  }
  return <span>{val}{suffix}</span>
}

export default function CompareExperimentsView({
  experiment,
  onViewBaseline,
  onViewProposed,
  onDownloadArchive,
  onExportBranch,
}) {
  const [showBasePaperModal, setShowBasePaperModal] = useState(false)

  if (!experiment) return null

  const {
    experiment_id,
    dataset_id,
    objective,
    created_at,
    baseline,
    proposed,
    comparison,
    persisted,
    shared_plan,
  } = experiment

  const deltaScore = comparison?.overall_score_delta ?? 0
  const isPositive = deltaScore > 0
  const isNeutral = deltaScore === 0

  // Combine dimension keys across both branches
  const dimensionKeys = Array.from(
    new Set([
      ...Object.keys(baseline?.dimension_scores || {}),
      ...Object.keys(proposed?.dimension_scores || {}),
      ...Object.keys(comparison?.dimension_deltas || {}),
    ])
  )

  // Granular findings measures with honest missing data handling
  const initialFindings =
    comparison?.initial_findings_count ??
    baseline?.total_tasks_executed ??
    shared_plan?.analysis_tasks?.length ??
    null

  const acceptedDirectly = comparison?.findings_accepted_directly ?? null
  const requiringReinv =
    comparison?.findings_requiring_reinvestigation ??
    comparison?.assumption_violations_flagged ??
    null
  const revisedFindings =
    comparison?.findings_revised ??
    comparison?.violations_resolved_via_reinvestigation ??
    null
  const unresolvedFindings =
    comparison?.findings_unresolved ??
    (comparison?.questionable_findings_count !== undefined && comparison?.questionable_findings_count !== null
      ? comparison.questionable_findings_count + (comparison?.rejected_findings_count || 0)
      : null)

  return (
    <div className="compare-experiments-container fade-in">
      {/* 1. Scientific Verdict Banner */}
      <div className="card verdict-card card-no-hover">
        <div className="verdict-header">
          <div className="verdict-badge">
            <span>🔬 Scientific Benchmark Verdict</span>
          </div>
          <div className="verdict-timestamp">
            {created_at ? new Date(created_at).toLocaleString() : ''}
          </div>
        </div>
        <div className="verdict-statement">
          {comparison?.scientific_verdict || 'Comparative evaluation complete.'}
        </div>
        <div className="verdict-meta-row">
          <span>Experiment ID: <code>{experiment_id}</code></span>
          <span>·</span>
          <span>Dataset: <strong>{dataset_id}</strong></span>
          <span>·</span>
          <span>Persistence: {persisted ? '✓ Saved to Disk' : 'In-Memory Only'}</span>
        </div>
        <div className="verdict-objective-row">
          <span className="objective-label">Objective:</span>
          <span className="objective-text">
            {objective && typeof objective === 'string' && objective.trim()
              ? objective.trim()
              : 'Objective not available'}
          </span>
        </div>
        <div className="verdict-actions-row" style={{ display: 'flex', gap: '10px', marginTop: '14px', flexWrap: 'wrap' }}>
          {onDownloadArchive && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={onDownloadArchive}
              title="Download complete research archive with Baseline, Proposed, Manifest, and SHA-256 Checksums (.ZIP)"
              style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', borderColor: '#818cf8', fontWeight: 600 }}
            >
              <span>📦 Download Complete Pair Archive (.ZIP)</span>
            </button>
          )}
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => setShowBasePaperModal(true)}
            title="Compare against published benchmarks (Weng et al., 2026)"
          >
            📊 Compare with Base Paper
          </button>
        </div>
      </div>

      {/* 2. Head-to-Head Branch Scorecard */}
      <div className="branches-grid">
        {/* Baseline Card */}
        <div className="card branch-card baseline-branch card-no-hover">
          <div className="branch-card-header">
            <div className="branch-tag baseline-tag">
              <span>🔬 Baseline Architecture</span>
            </div>
            <div className="branch-grade-pill grade-baseline">
              {baseline?.overall_grade || '—'}
            </div>
          </div>

          <div className="branch-score-display">
            <span className="branch-score-number">{fmt(baseline?.overall_score)}</span>
            <span className="branch-score-denom">/ 100</span>
          </div>
          <div className="branch-score-label">Overall Scientific Quality Score</div>

          <div className="branch-details-list">
            <div className="branch-detail-row">
              <span className="detail-name">Architecture</span>
              <span className="detail-value">LLM Plan + SciPy + Eval</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Critic Review</span>
              <span className="detail-value detail-inactive">Bypassed (None)</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Re-investigation</span>
              <span className="detail-value detail-inactive">Bypassed (None)</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Insight Synthesis</span>
              <span className="detail-value detail-inactive">Bypassed (None)</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Tasks Executed</span>
              <span className="detail-value">{baseline?.total_tasks_executed ?? 0}</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Run Identifier</span>
              <span className="detail-value detail-mono">{baseline?.run_id || '—'}</span>
            </div>
          </div>

          {onViewBaseline && (
            <button
              type="button"
              className="btn btn-secondary branch-inspect-btn"
              onClick={onViewBaseline}
            >
              <span>Inspect Baseline Pipeline</span>
              <span>→</span>
            </button>
          )}

          {onExportBranch && (
            <div style={{ display: 'flex', gap: '6px', marginTop: '10px' }}>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                style={{ flex: 1, fontSize: '0.75rem', padding: '4px 6px' }}
                onClick={() => onExportBranch('baseline', 'markdown')}
                title="Download Baseline Markdown report"
              >
                📥 Baseline .MD
              </button>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                style={{ flex: 1, fontSize: '0.75rem', padding: '4px 6px' }}
                onClick={() => onExportBranch('baseline', 'json')}
                title="Download Baseline JSON data"
              >
                { } Baseline JSON
              </button>
            </div>
          )}
        </div>

        {/* Proposed Card */}
        <div className="card branch-card proposed-branch card-no-hover">
          <div className="branch-card-header">
            <div className="branch-tag proposed-tag">
              <span>🛡 Proposed Architecture</span>
            </div>
            <div className="branch-grade-pill grade-proposed">
              {proposed?.overall_grade || '—'}
            </div>
          </div>

          <div className="branch-score-display">
            <span className="branch-score-number proposed-accent">{fmt(proposed?.overall_score)}</span>
            <span className="branch-score-denom">/ 100</span>
          </div>
          <div className="branch-score-label">Overall Scientific Quality Score</div>

          <div className="branch-details-list">
            <div className="branch-detail-row">
              <span className="detail-name">Architecture</span>
              <span className="detail-value">Plan + SciPy + Critic + Verif + Insights</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Critic Review</span>
              <span className="detail-value detail-active">Active (Audited)</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Re-investigation</span>
              <span className="detail-value detail-active">Active (Non-parametric)</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Insight Synthesis</span>
              <span className="detail-value detail-active">Active (Grounded)</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Tasks Executed</span>
              <span className="detail-value">{proposed?.total_tasks_executed ?? 0}</span>
            </div>
            <div className="branch-detail-row">
              <span className="detail-name">Run Identifier</span>
              <span className="detail-value detail-mono">{proposed?.run_id || '—'}</span>
            </div>
          </div>

          {onViewProposed && (
            <button
              type="button"
              className="btn btn-primary branch-inspect-btn"
              onClick={onViewProposed}
            >
              <span>Inspect Proposed Pipeline</span>
              <span>→</span>
            </button>
          )}

          {onExportBranch && (
            <div style={{ display: 'flex', gap: '6px', marginTop: '10px' }}>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                style={{ flex: 1, fontSize: '0.75rem', padding: '4px 6px' }}
                onClick={() => onExportBranch('proposed', 'markdown')}
                title="Download Proposed Markdown report"
              >
                📥 Proposed .MD
              </button>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                style={{ flex: 1, fontSize: '0.75rem', padding: '4px 6px' }}
                onClick={() => onExportBranch('proposed', 'json')}
                title="Download Proposed JSON data"
              >
                { } Proposed JSON
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 3. Visual Findings Lifecycle & Verification Comparison Chart */}
      <div className="card card-no-hover comparison-section-card">
        <div className="section-title">
          <span>Visual Findings Lifecycle & Verification Comparison</span>
          <span className="section-subtitle-tag">Preserving distinction between initial findings and final verified findings</span>
        </div>

        <div className="findings-comparison-grid">
          {/* Baseline Flow Column */}
          <div className="findings-flow-card baseline-flow">
            <div className="flow-card-header">
              <span className="flow-tag baseline-tag">🔬 Baseline Lifecycle</span>
              <span className="flow-status-pill">Un-critiqued Pass-through</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Initial Findings Generated:</span>
              <span className="flow-stat-val">{renderValueOrNotRecorded(initialFindings)}</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Critic Assumption Audits:</span>
              <span className="flow-stat-val text-muted">0 (Bypassed)</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Un-critiqued Findings Retained:</span>
              <span className="flow-stat-val text-warning">
                {initialFindings !== null ? `${initialFindings} (100%)` : renderValueOrNotRecorded(null)}
              </span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Alternative Re-investigations:</span>
              <span className="flow-stat-val text-muted">0 (None)</span>
            </div>
            <div className="flow-summary-box baseline-box">
              All initial statistical findings are accepted directly without parametric assumption verification or outlier leverage filtering.
            </div>
          </div>

          {/* Proposed Flow Column */}
          <div className="findings-flow-card proposed-flow">
            <div className="flow-card-header">
              <span className="flow-tag proposed-tag">🛡 Proposed Critic Lifecycle</span>
              <span className="flow-status-pill pill-verified">Critic-Audited Lifecycle</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Initial Findings Evaluated:</span>
              <span className="flow-stat-val">{renderValueOrNotRecorded(initialFindings)}</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Findings Accepted Directly:</span>
              <span className="flow-stat-val text-positive">{renderValueOrNotRecorded(acceptedDirectly)}</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Requiring Re-investigation:</span>
              <span className="flow-stat-val text-warning">{renderValueOrNotRecorded(requiringReinv)}</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Revised via Robust Re-test:</span>
              <span className="flow-stat-val text-info">{renderValueOrNotRecorded(revisedFindings)}</span>
            </div>
            <div className="flow-stat-row">
              <span className="flow-stat-label">Unresolved (Caveats/Rejected):</span>
              <span className="flow-stat-val text-danger">{renderValueOrNotRecorded(unresolvedFindings)}</span>
            </div>
            <div className="flow-summary-box proposed-box">
              Critic Agent audits Cochran's rule, normality, and residual leverage outliers, dynamically swapping failed tests for robust non-parametric alternatives.
            </div>
          </div>
        </div>

        {/* Visual Findings Distribution Bar */}
        <div className="findings-visual-bar-wrap">
          <div className="visual-bar-header">
            <span>Proposed Finding Verdict Distribution</span>
            <span className="visual-bar-sub">Empirically audited distribution</span>
          </div>
          <div className="visual-proportional-bar">
            {comparison?.verified_findings_count ? (
              <div
                className="bar-segment seg-verified"
                style={{
                  flexGrow: comparison.verified_findings_count,
                  minWidth: '40px',
                }}
                title={`Verified: ${comparison.verified_findings_count}`}
              >
                <span>Verified ({comparison.verified_findings_count})</span>
              </div>
            ) : null}
            {comparison?.questionable_findings_count ? (
              <div
                className="bar-segment seg-questionable"
                style={{
                  flexGrow: comparison.questionable_findings_count,
                  minWidth: '40px',
                }}
                title={`Questionable: ${comparison.questionable_findings_count}`}
              >
                <span>Caveats ({comparison.questionable_findings_count})</span>
              </div>
            ) : null}
            {comparison?.rejected_findings_count ? (
              <div
                className="bar-segment seg-rejected"
                style={{
                  flexGrow: comparison.rejected_findings_count,
                  minWidth: '30px',
                }}
                title={`Rejected: ${comparison.rejected_findings_count}`}
              >
                <span>Rejected ({comparison.rejected_findings_count})</span>
              </div>
            ) : null}
          </div>
        </div>

        {/* Scientific Integrity Note on Acceptance Rate */}
        <div className="acceptance-disclaimer-banner">
          <span className="disclaimer-icon">ℹ️</span>
          <div className="disclaimer-text">
            <strong>Scientific Integrity Note on Acceptance:</strong> The Critic accepted the finding under the implemented verification checks. Acceptance does not establish statistical correctness, causal validity, or mathematical proof.
          </div>
        </div>
      </div>

      {/* 4. Empirical Delta Metrics Summary */}
      <div className="card card-no-hover comparison-section-card">
        <div className="section-title">
          <span>Empirical Delta Comparison</span>
          <span className="section-subtitle-tag">Strictly derived from actual service evaluations</span>
        </div>

        <div className="metrics-delta-grid">
          <div className="delta-stat-card">
            <div className="delta-stat-title">Overall Score Delta</div>
            <div className={`delta-stat-value ${isPositive ? 'delta-stat-pos' : isNeutral ? 'delta-stat-neu' : 'delta-stat-neg'}`}>
              {signedFmt(comparison?.overall_score_delta)} pts
            </div>
            <div className="delta-stat-sub">
              {fmt(baseline?.overall_score)} → {fmt(proposed?.overall_score)}
            </div>
          </div>

          <div className="delta-stat-card">
            <div className="delta-stat-title">Statistical Rigor Delta</div>
            <div className={`delta-stat-value ${(comparison?.rigor_score_delta ?? 0) >= 0 ? 'delta-stat-pos' : 'delta-stat-neg'}`}>
              {signedFmt(comparison?.rigor_score_delta)} pts
            </div>
            <div className="delta-stat-sub">Assumption audits & validity</div>
          </div>

          <div className="delta-stat-card">
            <div className="delta-stat-title">Verification Grounding Delta</div>
            <div className={`delta-stat-value ${(comparison?.verification_score_delta ?? 0) >= 0 ? 'delta-stat-pos' : 'delta-stat-neg'}`}>
              {signedFmt(comparison?.verification_score_delta)} pts
            </div>
            <div className="delta-stat-sub">Empirical truth traceability</div>
          </div>

          <div className="delta-stat-card">
            <div className="delta-stat-title">Re-investigation Loops</div>
            <div className="delta-stat-value delta-stat-neutral">
              {renderValueOrNotRecorded(comparison?.total_reinvestigations)}
            </div>
            <div className="delta-stat-sub">Controlled non-parametric tests</div>
          </div>
        </div>

        {/* Detailed Dimension Deltas Table */}
        {dimensionKeys.length > 0 && (
          <div className="dimension-table-wrap">
            <table className="dimension-table">
              <thead>
                <tr>
                  <th>Evaluation Dimension</th>
                  <th style={{ textAlign: 'right' }}>Baseline Score</th>
                  <th style={{ textAlign: 'right' }}>Proposed Score</th>
                  <th style={{ textAlign: 'right' }}>Net Delta</th>
                </tr>
              </thead>
              <tbody>
                {dimensionKeys.map((key) => {
                  const bVal = baseline?.dimension_scores?.[key]
                  const pVal = proposed?.dimension_scores?.[key]
                  const dVal = comparison?.dimension_deltas?.[key] ?? (
                    bVal !== undefined && pVal !== undefined ? pVal - bVal : null
                  )
                  const deltaClass = dVal > 0 ? 'text-positive' : dVal < 0 ? 'text-negative' : 'text-neutral'

                  return (
                    <tr key={key}>
                      <td className="dimension-name-cell">
                        <strong>{formatDimensionName(key)}</strong>
                      </td>
                      <td style={{ textAlign: 'right' }}>{fmt(bVal)}</td>
                      <td style={{ textAlign: 'right' }}>{fmt(pVal)}</td>
                      <td style={{ textAlign: 'right' }} className={deltaClass}>
                        <strong>{signedFmt(dVal)}</strong>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 5. Critic Auditing & Governance Summary */}
      <div className="card card-no-hover comparison-section-card">
        <div className="section-title">
          <span>Critic Auditing & Verification Governance</span>
          <span className="section-subtitle-tag">Empirical protection against false positives & unverified claims</span>
        </div>

        <div className="audit-metrics-grid">
          <div className="audit-card">
            <div className="audit-number text-warning">
              {renderValueOrNotRecorded(comparison?.assumption_violations_flagged)}
            </div>
            <div className="audit-label">Assumption Violations Flagged</div>
            <div className="audit-desc">
              Findings identified with normality failures, extreme outliers, or inadequate sample sizes.
            </div>
          </div>

          <div className="audit-card">
            <div className="audit-number text-positive">
              {renderValueOrNotRecorded(comparison?.violations_resolved_via_reinvestigation)}
            </div>
            <div className="audit-label">Violations Resolved via Re-test</div>
            <div className="audit-desc">
              Flagged hypotheses successfully resolved with robust non-parametric alternatives.
            </div>
          </div>

          <div className="audit-card">
            <div className="audit-number text-info">
              {comparison?.resolution_rate !== null && comparison?.resolution_rate !== undefined
                ? `${(comparison.resolution_rate * 100).toFixed(1)}%`
                : <span className="text-muted-italic">Not recorded</span>}
            </div>
            <div className="audit-label">Audit Resolution Rate</div>
            <div className="audit-desc">
              Proportion of flagged methodological violations successfully recovered.
            </div>
          </div>
        </div>

        {/* Findings Classification Breakdown */}
        <div className="findings-status-bar">
          <div className="findings-status-item status-verified">
            <span className="status-dot dot-verified"></span>
            <span className="status-label">Accepted Under Audit:</span>
            <strong>{renderValueOrNotRecorded(comparison?.verified_findings_count)}</strong>
          </div>
          <div className="findings-status-item status-questionable">
            <span className="status-dot dot-questionable"></span>
            <span className="status-label">Questionable (Caveats):</span>
            <strong>{renderValueOrNotRecorded(comparison?.questionable_findings_count)}</strong>
          </div>
          <div className="findings-status-item status-rejected">
            <span className="status-dot dot-rejected"></span>
            <span className="status-label">Methodologically Rejected:</span>
            <strong>{renderValueOrNotRecorded(comparison?.rejected_findings_count)}</strong>
          </div>
        </div>
      </div>

      {/* 6. Base-Paper Comparison & Methodological Safeguards */}
      <div className="card card-no-hover safeguards-card">
        <div className="safeguards-header">
          <span>📄 Base-Paper Reference & Benchmark Lineage</span>
        </div>
        <div className="safeguards-body">
          <p>
            Benchmarked against the research foundation: <strong>"AutoMind: Automated Insight Discovery via Multi-Agent Navigation"</strong>, <em>IEEE Access</em>, 2026, Vol. 14, pp. 23936–23955, DOI: 10.1109/ACCESS.2026.3661202.
          </p>
          <div className="base-paper-btn-wrap" style={{ marginTop: '12px' }}>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => setShowBasePaperModal(true)}
            >
              <span>📄 View Base-Paper Feature Comparison & Lineage Matrix</span>
            </button>
          </div>
        </div>
      </div>

      {/* Base-Paper Reference Modal */}
      <BasePaperComparisonModal
        isOpen={showBasePaperModal}
        onClose={() => setShowBasePaperModal(false)}
      />
    </div>
  )
}

