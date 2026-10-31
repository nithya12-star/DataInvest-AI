// components/PlanSection.jsx — Phase 2 Investigation Planner UI
import { useState } from 'react'
import './PlannerSection.css'

export default function PlanSection({
  profile,
  plan,
  objective,
  setObjective,
  loading,
  error,
  onGeneratePlan,
  onProceedToAnalyze,
  onRunComparativeExperiment,
  runningComparative = false,
}) {
  const [useLlm, setUseLlm] = useState(true)

  const formatRelation = (task) => {
    if (task.column_a && task.column_b) {
      return (
        <div className="task-columns-row">
          <span>{task.column_a}</span>
          <span className="task-relation-symbol">↔</span>
          <span>{task.column_b}</span>
        </div>
      )
    }
    if (task.group_column && task.value_column) {
      return (
        <div className="task-columns-row">
          <span>{task.group_column}</span>
          <span className="task-relation-symbol">→</span>
          <span>{task.value_column}</span>
        </div>
      )
    }
    if (task.column_a) {
      return (
        <div className="task-columns-row">
          <span>{task.column_a}</span>
        </div>
      )
    }
    return (
      <div className="task-columns-row">
        <span>{task.task}</span>
      </div>
    )
  }

  const formatTypeLabel = (type) => {
    switch (type) {
      case 'numerical_vs_numerical': return 'Bivariate Correlation'
      case 'two_groups_vs_numerical': return '2-Group Comparison'
      case 'multiple_groups_vs_numerical': return 'Multi-Group ANOVA'
      case 'categorical_vs_categorical': return 'Contingency Analysis'
      case 'outlier_sensitivity': return 'Outlier Sensitivity'
      case 'distribution_analysis': return 'Distribution Analysis'
      case 'missing_value_investigation': return 'Data Quality Audit'
      default: return type.replace(/_/g, ' ')
    }
  }

  return (
    <div className="plan-section-container fade-in">
      <div className="card card-no-hover section-card">
        <div className="section-title" style={{ marginBottom: '16px' }}>
          <span>Formulate Analytical Objective</span>
        </div>

        <form className="planner-form" onSubmit={(e) => onGeneratePlan(e, { use_llm: true })}>
          <div className="planner-input-wrap">
            <input
              type="text"
              className="planner-input"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="Ask what you want to analyze (e.g., 'Find factors affecting salary' or 'Analyze revenue drivers')..."
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary planner-btn"
            disabled={loading || !profile}
          >
            {loading ? (
              <>
                <div className="planner-spinner"></div>
                <span>Generating plan...</span>
              </>
            ) : (
              <>
                <span>⚡</span>
                <span>{plan ? 'Regenerate Plan' : 'Generate Analysis Plan'}</span>
              </>
            )}
          </button>
          <div className="planner-options-row" style={{ width: '100%', marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: '#94a3b8' }}>
              <span style={{ color: '#10b981', fontWeight: 600 }}>● Active</span>
              <span>Planner Engine: Google Gemini <code>gemini-3.5-flash-lite</code> (LLM Enforced)</span>
            </span>
          </div>
        </form>

        {error && (
          <div
            className="planner-warnings"
            style={{
              borderColor: 'var(--error)',
              background: 'rgba(239, 68, 68, 0.08)',
              marginTop: '12px',
            }}
          >
            <div className="warnings-header" style={{ color: 'var(--error)' }}>
              <span>⚠</span>
              <span>Planning Error</span>
            </div>
            <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
              {error}
            </div>
          </div>
        )}
      </div>

      {plan && (
        <div className="plan-results fade-in" style={{ marginTop: '20px' }}>
          {/* Interpreted Goal & Target */}
          <div className="plan-overview-card">
            <div className="plan-goal-title">Interpreted Analytical Objective</div>
            <div className="plan-goal-text">{plan.interpreted_goal}</div>

            {/* Truthful Planner Execution Badge */}
            <div className="planner-metadata-badge" style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              {plan.planner_mode === 'llm' ? (
                <span className="badge badge-purple" style={{ padding: '4px 10px', fontSize: '0.78rem' }} title="Verified LLM-generated plan">
                  🤖 Planner: LLM | Model: {plan.planner_model || plan.dataset_summary?.model || 'gemini-3.5-flash-lite'} | Provider: {plan.planner_provider || 'Google'}
                </span>
              ) : plan.planner_mode === 'deterministic' ? (
                <span className="badge badge-secondary" style={{ padding: '4px 10px', fontSize: '0.78rem' }} title="Rule-based heuristic plan">
                  ⚙ Planner: Deterministic (rule-based)
                </span>
              ) : (
                <span className="badge badge-warning" style={{ padding: '4px 10px', fontSize: '0.78rem' }}>
                  ℹ Planner: Unavailable
                </span>
              )}
            </div>

            {plan.target_column && (
              <div className="target-badge" style={{ marginTop: '12px' }}>
                <span>🎯 Target Variable:</span>
                <strong>{plan.target_column}</strong>
              </div>
            )}

            {/* Selected Variables */}
            {plan.selected_columns?.length > 0 && (
              <div className="selected-vars-wrap">
                <div className="selected-vars-label">
                  Selected Variables ({plan.selected_columns.length})
                </div>
                <div className="selected-chips">
                  {plan.selected_columns.map((sc, idx) => (
                    <div
                      key={idx}
                      className={`var-chip ${sc.is_target ? 'is-target' : ''}`}
                      title={sc.reason}
                    >
                      <span>{sc.column}</span>
                      <span className="var-type">({sc.data_type})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Warnings Banner */}
          {plan.warnings?.length > 0 && (
            <div className="planner-warnings">
              <div className="warnings-header">
                <span>⚠</span>
                <span>Data Quality Considerations</span>
              </div>
              {plan.warnings.map((w, idx) => (
                <div key={idx} className="warning-item">
                  • {w}
                </div>
              ))}
            </div>
          )}

          {/* Planned Tasks Header */}
          <div className="section-title" style={{ marginTop: '10px', marginBottom: '8px' }}>
            <span>Planned Investigation Tasks ({plan.analysis_tasks?.length || 0})</span>
          </div>

          {/* Planned Tasks Grid */}
          <div className="tasks-grid">
            {plan.analysis_tasks?.map((t) => (
              <div key={t.task_id} className="task-card">
                <div className="task-top-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="task-id-badge">{t.task_id}</span>
                    <span className="task-type-badge">{formatTypeLabel(t.task_type)}</span>
                  </div>
                  <span className={`priority-badge priority-${t.priority}`}>
                    {t.priority} priority
                  </span>
                </div>

                {formatRelation(t)}

                <div className="task-desc">{t.description}</div>

                <div className="task-bottom-row">
                  <div className="method-tag">
                    <span>🔬 Method:</span>
                    <span>{t.method}</span>
                  </div>
                  <div className="task-reason">Rationale: {t.reason}</div>
                </div>

                {t.notes && t.notes.length > 0 && (
                  <div className="task-notes">
                    {t.notes.map((note, nIdx) => (
                      <div key={nIdx}>ℹ {note}</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Bottom Call-to-Action to proceed to Phase 3 or Run Comparative Experiment */}
          {onProceedToAnalyze && (
            <div className="phase-transition-cta card card-no-hover fade-in">
              <div className="cta-content">
                <div className="cta-title">Investigation Plan Ready</div>
                <div className="cta-subtitle">
                  {plan.analysis_tasks.length} prioritized tasks ready for execution. Proceed step-by-step or run a head-to-head Baseline vs Proposed benchmark.
                </div>
              </div>
              <div className="cta-buttons-group" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={onProceedToAnalyze}
                >
                  <span>Proceed to Statistical Analysis (Phase 3)</span>
                  <span>→</span>
                </button>

                {onRunComparativeExperiment && (
                  <button
                    type="button"
                    className="btn btn-secondary phase-action-btn"
                    onClick={onRunComparativeExperiment}
                    disabled={runningComparative}
                    title="Execute paired experiment benchmark (Baseline vs Proposed)"
                    style={{ borderColor: 'rgba(99, 102, 241, 0.5)', background: 'rgba(99, 102, 241, 0.12)' }}
                  >
                    {runningComparative ? (
                      <>
                        <div className="planner-spinner"></div>
                        <span>Benchmarking Pipelines...</span>
                      </>
                    ) : (
                      <>
                        <span>⚖</span>
                        <span>Run Comparative Experiment (Baseline vs Proposed)</span>
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
