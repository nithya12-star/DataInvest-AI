// frontend/src/components/ExperimentHistoryModal.jsx — Experiment Run Repository UI
import { useState } from 'react'
import { downloadRunExport } from '../services/api'
import './ExperimentHistoryModal.css'

export default function ExperimentHistoryModal({
  isOpen,
  onClose,
  runs = [],
  activeRunId = null,
  onOpenRun,
  onDeleteRun,
  onNewUpload,
}) {
  const [deletingId, setDeletingId] = useState(null)
  const [openingId, setOpeningId] = useState(null)
  const [exportingId, setExportingId] = useState(null)
  const [exportError, setExportError] = useState('')

  if (!isOpen) return null

  const handleOpen = async (runId) => {
    setOpeningId(runId)
    try {
      await onOpenRun(runId)
      onClose()
    } finally {
      setOpeningId(null)
    }
  }

  const handleDelete = async (runId, datasetName) => {
    if (!window.confirm(`Are you sure you want to delete the experiment run for "${datasetName}"?`)) {
      return
    }
    setDeletingId(runId)
    try {
      await onDeleteRun(runId)
    } finally {
      setDeletingId(null)
    }
  }

  const handleExport = async (runId, format) => {
    setExportingId(`${runId}_${format}`)
    setExportError('')
    try {
      await downloadRunExport(runId, format)
    } catch (err) {
      setExportError(err.message || `Failed to export ${format}`)
      console.error('Export error:', err)
    } finally {
      setExportingId(null)
    }
  }

  const formatTimestamp = (isoStr) => {
    if (!isoStr) return '—'
    try {
      const d = new Date(isoStr)
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return isoStr
    }
  }

  const phaseLabels = [
    { id: 1, label: 'Profile' },
    { id: 2, label: 'Plan' },
    { id: 3, label: 'Analyze' },
    { id: 4, label: 'Critic' },
    { id: 5, label: 'Verify' },
    { id: 6, label: 'Insights' },
    { id: 7, label: 'Evaluate' },
  ]

  return (
    <div className="history-modal-overlay" onClick={onClose}>
      <div className="history-modal" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="history-modal-header">
          <div className="history-modal-title-wrap">
            <span className="history-modal-title">
              <span>📁</span>
              <span>Experiment History</span>
            </span>
            <span className="badge badge-cyan">{runs.length} Runs Saved</span>
          </div>
          <button
            type="button"
            className="history-modal-close"
            onClick={onClose}
            aria-label="Close History"
          >
            ×
          </button>
        </div>

        {/* Modal Body */}
        <div className="history-modal-body">
          {exportError && (
            <div
              className="history-export-error-banner fade-in"
              role="alert"
              style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                color: '#fca5a5',
                padding: '10px 14px',
                borderRadius: '8px',
                marginBottom: '14px',
                fontSize: '0.85rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '12px',
              }}
            >
              <span>⚠ {exportError}</span>
              <button
                type="button"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#fca5a5',
                  cursor: 'pointer',
                  fontSize: '1.2rem',
                  lineHeight: 1,
                  padding: '0 4px',
                }}
                onClick={() => setExportError('')}
                aria-label="Dismiss export error"
              >
                ×
              </button>
            </div>
          )}

          {runs.length === 0 ? (
            <div className="history-empty-state">
              <div className="history-empty-icon">📭</div>
              <h4>No Saved Experiments Yet</h4>
              <p className="history-empty-desc">
                When you upload a dataset and execute pipeline phases, each run is automatically
                persisted and will appear here.
              </p>
            </div>
          ) : (
            runs.map((run) => {
              const isActive = run.run_id === activeRunId
              const isDeleting = deletingId === run.run_id
              const isOpening = openingId === run.run_id
              const completed = run.completed_phases || []

              return (
                <div
                  key={run.run_id}
                  className={`history-card ${isActive ? 'active-run' : ''}`}
                >
                  <div className="history-card-main">
                    <div className="history-card-header">
                      <span className="history-card-name">{run.dataset_name}</span>
                      {run.mode && (
                        <span
                          className={`badge ${run.mode.toLowerCase() === 'proposed' ? 'badge-purple' : 'badge-warning'}`}
                          style={{ fontSize: '0.75rem', textTransform: 'uppercase' }}
                          title={`Execution Mode: ${run.mode}`}
                        >
                          {run.mode}
                        </span>
                      )}
                      {run.planner_mode === 'llm' && (
                        <span
                          className="badge badge-purple"
                          style={{ fontSize: '0.75rem' }}
                          title={`Planner: LLM (${run.planner_model || 'gemini-3.5-flash-lite'})`}
                        >
                          🤖 LLM: {run.planner_model || 'gemini'}
                        </span>
                      )}
                      {run.planner_mode === 'deterministic' && (
                        <span
                          className="badge badge-secondary"
                          style={{ fontSize: '0.75rem' }}
                          title="Planner: Deterministic (rule-based)"
                        >
                          ⚙ Deterministic
                        </span>
                      )}
                      {run.has_visual_evidence && (
                        <span
                          className="badge badge-indigo"
                          style={{ fontSize: '0.75rem' }}
                          title="Contains interactive Vega-Lite visual evidence"
                        >
                          📊 Charts
                        </span>
                      )}
                      {isActive && (
                        <span className="badge badge-green" style={{ fontSize: '0.75rem' }}>
                          ● Active
                        </span>
                      )}
                      {run.overall_score !== null && run.overall_score !== undefined && (
                        <span className="badge badge-cyan" style={{ fontSize: '0.75rem' }}>
                          Score: {run.overall_score} / 100 ({run.overall_grade || 'Grade'})
                        </span>
                      )}
                    </div>

                    <div className="history-card-meta">
                      <span>Updated: {formatTimestamp(run.updated_at || run.created_at)}</span>
                      {run.n_rows && (
                        <span>
                          • {run.n_rows.toLocaleString()} rows × {run.n_cols} cols
                        </span>
                      )}
                      <span style={{ opacity: 0.6, fontSize: '0.75rem' }}>ID: {run.run_id}</span>
                    </div>

                    {/* Completed Phases Progress Indicators */}
                    <div className="history-phases-wrap">
                      {phaseLabels.map((p) => (
                        <span
                          key={p.id}
                          className={`phase-pill ${completed.includes(p.id) ? 'completed' : ''}`}
                          title={`Phase ${p.id}: ${p.label} ${completed.includes(p.id) ? 'Completed' : 'Pending'}`}
                        >
                          {completed.includes(p.id) ? '✓' : ''} P{p.id} {p.label}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="history-card-actions">
                    <button
                      type="button"
                      className="btn-export-run"
                      onClick={() => handleExport(run.run_id, 'markdown')}
                      disabled={isOpening || isDeleting || exportingId === `${run.run_id}_markdown`}
                      title="Download readable Markdown research report for this run"
                    >
                      {exportingId === `${run.run_id}_markdown` ? '...' : '📥 .MD'}
                    </button>
                    <button
                      type="button"
                      className="btn-export-run"
                      onClick={() => handleExport(run.run_id, 'json')}
                      disabled={isOpening || isDeleting || exportingId === `${run.run_id}_json`}
                      title="Download structured JSON run state & report"
                    >
                      {exportingId === `${run.run_id}_json` ? '...' : '{ } JSON'}
                    </button>
                    <button
                      type="button"
                      className="btn-open-run"
                      onClick={() => handleOpen(run.run_id)}
                      disabled={isOpening || isDeleting}
                      title="Load this run into workspace without re-running computations"
                    >
                      {isOpening ? 'Loading...' : isActive ? 'Viewing' : 'Open Run →'}
                    </button>
                    <button
                      type="button"
                      className="btn-delete-run"
                      onClick={() => handleDelete(run.run_id, run.dataset_name)}
                      disabled={isOpening || isDeleting}
                      title="Delete this saved run"
                    >
                      {isDeleting ? '...' : '🗑'}
                    </button>
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* Modal Footer */}
        <div className="history-modal-footer">
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => {
              onClose()
              if (onNewUpload) onNewUpload()
            }}
          >
            ↑ Upload New Dataset
          </button>
          <button type="button" className="btn btn-outline btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
