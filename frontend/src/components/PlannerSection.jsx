// components/PlannerSection.jsx — Phase 2 Planner, Phase 3 Analysis, Phase 4 Critic, & Phase 5 Verified Findings UI
import { useState } from 'react'
import {
  createInvestigationPlan,
  executeAnalysisPlan,
  runCriticReview,
  verifyFindings,
} from '../services/api'
import AnalysisResults from './AnalysisResults'
import CriticSection from './CriticSection'
import VerifiedFindingsSection from './VerifiedFindingsSection'
import './PlannerSection.css'

export default function PlannerSection({ profile }) {
  const [objective, setObjective] = useState('Investigate this dataset.')
  const [useLlm, setUseLlm] = useState(true)
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Phase 3 Statistical Execution state
  const [analysisResponse, setAnalysisResponse] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState('')

  // Phase 4 Critic Verification state
  const [criticResponse, setCriticResponse] = useState(null)
  const [critiquing, setCritiquing] = useState(false)
  const [criticError, setCriticError] = useState('')

  // Phase 5 Verified Findings & Re-investigation state
  const [verificationResponse, setVerificationResponse] = useState(null)
  const [verifying, setVerifying] = useState(false)
  const [verificationError, setVerificationError] = useState('')

  const handleGeneratePlan = async (e) => {
    e?.preventDefault()
    if (!profile) return

    setLoading(true)
    setError('')
    setAnalysisResponse(null)
    setAnalysisError('')
    setCriticResponse(null)
    setCriticError('')
    setVerificationResponse(null)
    setVerificationError('')

    try {
      const planResponse = await createInvestigationPlan(profile, objective, { use_llm: useLlm })
      setPlan(planResponse)
    } catch (err) {
      setError(err.message || 'Unable to create an investigation plan.')
    } finally {
      setLoading(false)
    }
  }

  const handleRunAnalysis = async () => {
    if (!plan || !plan.analysis_tasks || plan.analysis_tasks.length === 0) return

    setAnalyzing(true)
    setAnalysisError('')
    setCriticResponse(null)
    setCriticError('')
    setVerificationResponse(null)
    setVerificationError('')

    try {
      const datasetId = profile?.dataset_id || profile?.filename || 'sample.csv'
      const response = await executeAnalysisPlan(datasetId, plan.analysis_tasks)
      setAnalysisResponse(response)
    } catch (err) {
      setAnalysisError(err.message || 'Failed to execute statistical analysis plan.')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleRunCritic = async () => {
    if (!analysisResponse) return

    setCritiquing(true)
    setCriticError('')
    setVerificationResponse(null)
    setVerificationError('')

    try {
      const response = await runCriticReview(analysisResponse, profile, plan)
      setCriticResponse(response)
    } catch (err) {
      setCriticError(err.message || 'Failed to run Critic verification review.')
    } finally {
      setCritiquing(false)
    }
  }

  const handleVerifyFindings = async () => {
    if (!analysisResponse || !criticResponse) return

    setVerifying(true)
    setVerificationError('')

    try {
      const response = await verifyFindings(analysisResponse, criticResponse, profile, plan)
      setVerificationResponse(response)
    } catch (err) {
      setVerificationError(
        err.message || 'Failed to execute Phase 5 verification and re-investigation.'
      )
    } finally {
      setVerifying(false)
    }
  }

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
    <div className="planner-section card card-no-hover section-card fade-in">
      <div className="planner-header">
        <div className="planner-title-wrap">
          <div className="section-title" style={{ marginBottom: 0 }}>
            <span>Analytical Pipeline: Planner, Execution, Critic & Verified Findings</span>
          </div>
          <span className="badge badge-cyan">Phases 2, 3, 4 & 5 Active</span>
        </div>
      </div>

      <form className="planner-form" onSubmit={handleGeneratePlan}>
        <div className="planner-input-wrap">
          <input
            type="text"
            className="planner-input"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            placeholder="Ask what you want to analyze (e.g., 'Find factors affecting salary')..."
            disabled={loading || analyzing || critiquing || verifying}
          />
        </div>
        <button
          type="submit"
          className="btn btn-primary planner-btn"
          disabled={loading || analyzing || critiquing || verifying || !profile}
        >
          {loading ? (
            <>
              <div className="planner-spinner"></div>
              <span>Creating plan...</span>
            </>
          ) : (
            <>
              <span>⚡</span>
              <span>Generate Analysis Plan</span>
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
        <div className="planner-warnings" style={{ borderColor: 'var(--error)', background: 'rgba(239, 68, 68, 0.08)', marginBottom: '16px' }}>
          <div className="warnings-header" style={{ color: 'var(--error)' }}>
            <span>⚠</span>
            <span>Planning Error</span>
          </div>
          <div className="warning-item" style={{ color: 'var(--text-primary)' }}>{error}</div>
        </div>
      )}

      {plan && (
        <div className="plan-results fade-in">
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
                <div className="selected-vars-label">Selected Variables ({plan.selected_columns.length})</div>
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
                <div key={idx} className="warning-item">• {w}</div>
              ))}
            </div>
          )}

          {/* Planned Tasks Header & Action Button */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '16px', marginBottom: '12px', flexWrap: 'wrap', gap: '10px' }}>
            <div className="section-title" style={{ marginBottom: 0 }}>
              <span>Planned Investigation Tasks ({plan.analysis_tasks?.length || 0})</span>
            </div>

            <button
              className="btn btn-primary"
              style={{
                background: 'linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%)',
                boxShadow: '0 0 15px rgba(14, 165, 233, 0.4)',
                padding: '8px 18px',
                fontSize: '0.9rem',
                fontWeight: '600',
              }}
              onClick={handleRunAnalysis}
              disabled={analyzing || critiquing || verifying}
            >
              {analyzing ? (
                <>
                  <div className="planner-spinner"></div>
                  <span>Executing Statistical Tests (SciPy)...</span>
                </>
              ) : (
                <>
                  <span>🔬</span>
                  <span>Execute Statistical Analysis (Phase 3)</span>
                </>
              )}
            </button>
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

          {/* Execution Error Banner if any */}
          {analysisError && (
            <div className="planner-warnings" style={{ borderColor: 'var(--error)', background: 'rgba(239, 68, 68, 0.08)', marginTop: '20px' }}>
              <div className="warnings-header" style={{ color: 'var(--error)' }}>
                <span>⚠</span>
                <span>Analysis Execution Error</span>
              </div>
              <div className="warning-item" style={{ color: 'var(--text-primary)' }}>{analysisError}</div>
            </div>
          )}

          {/* Phase 3 Statistical Results Section */}
          {analysisResponse && (
            <div style={{ marginTop: '20px' }}>
              <AnalysisResults analysisResponse={analysisResponse} />

              {/* Action Button to trigger Phase 4 Critic Review */}
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '24px', marginBottom: '8px' }}>
                <button
                  className="btn btn-primary"
                  style={{
                    background: 'linear-gradient(135deg, #10b981 0%, #0ea5e9 100%)',
                    boxShadow: '0 0 20px rgba(16, 185, 129, 0.4)',
                    padding: '12px 28px',
                    fontSize: '1rem',
                    fontWeight: '700',
                  }}
                  onClick={handleRunCritic}
                  disabled={critiquing || verifying}
                >
                  {critiquing ? (
                    <>
                      <div className="planner-spinner"></div>
                      <span>Verifying Findings (Critic Agent)...</span>
                    </>
                  ) : (
                    <>
                      <span>🛡</span>
                      <span>Run Critic Verification Review (Phase 4)</span>
                    </>
                  )}
                </button>
              </div>

              {/* Critic Error Banner if any */}
              {criticError && (
                <div className="planner-warnings" style={{ borderColor: 'var(--error)', background: 'rgba(239, 68, 68, 0.08)', marginTop: '16px' }}>
                  <div className="warnings-header" style={{ color: 'var(--error)' }}>
                    <span>⚠</span>
                    <span>Critic Review Error</span>
                  </div>
                  <div className="warning-item" style={{ color: 'var(--text-primary)' }}>{criticError}</div>
                </div>
              )}

              {/* Phase 4 Critic Verification Results */}
              {criticResponse && (
                <>
                  <CriticSection criticResponse={criticResponse} />

                  {/* Action Button to trigger Phase 5 Re-investigation & Verified Findings */}
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'center',
                      marginTop: '28px',
                      marginBottom: '10px',
                    }}
                  >
                    <button
                      className="btn btn-primary"
                      style={{
                        background:
                          'linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%)',
                        boxShadow: '0 0 24px rgba(16, 185, 129, 0.45)',
                        padding: '14px 32px',
                        fontSize: '1.02rem',
                        fontWeight: '700',
                        borderRadius: 'var(--radius-md)',
                      }}
                      onClick={handleVerifyFindings}
                      disabled={verifying}
                    >
                      {verifying ? (
                        <>
                          <div className="planner-spinner"></div>
                          <span>Re-investigating & Verifying Findings (Phase 5)...</span>
                        </>
                      ) : (
                        <>
                          <span>✨</span>
                          <span>
                            Run Re-investigation & Synthesize Verified Findings (Phase 5)
                          </span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Phase 5 Error Banner if any */}
                  {verificationError && (
                    <div
                      className="planner-warnings"
                      style={{
                        borderColor: 'var(--error)',
                        background: 'rgba(239, 68, 68, 0.08)',
                        marginTop: '16px',
                      }}
                    >
                      <div className="warnings-header" style={{ color: 'var(--error)' }}>
                        <span>⚠</span>
                        <span>Finding Verification Error</span>
                      </div>
                      <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
                        {verificationError}
                      </div>
                    </div>
                  )}

                  {/* Phase 5 Verified Findings Results */}
                  {verificationResponse && (
                    <VerifiedFindingsSection verificationResponse={verificationResponse} />
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

