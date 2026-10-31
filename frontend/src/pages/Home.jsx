// pages/Home.jsx — Central Investigation Pipeline Orchestrator (Phases 1–6)
import { useState, useEffect } from 'react'
import './Home.css'
import MetricCard from '../components/MetricCard'
import DatasetPreview from '../components/DatasetPreview'
import ColumnProfile from '../components/ColumnProfile'
import NumericalProfile from '../components/NumericalProfile'
import DataQuality from '../components/DataQuality'
import InvestigationStepper from '../components/InvestigationStepper'
import CompletedPhaseSummary from '../components/CompletedPhaseSummary'
import PlanSection from '../components/PlanSection'
import AnalysisResults from '../components/AnalysisResults'
import CriticSection from '../components/CriticSection'
import VerifiedFindingsSection from '../components/VerifiedFindingsSection'
import InsightsSection from '../components/InsightsSection'
import EvaluationSection from '../components/EvaluationSection'
import ExperimentModeToggle from '../components/ExperimentModeToggle'
import CompareExperimentsView from '../components/CompareExperimentsView'
import { PHASES, getPhaseById } from '../constants/phases'
import {
  createInvestigationPlan,
  executeAnalysisPlan,
  runCriticReview,
  verifyFindings,
  generateInsights,
  evaluatePipeline,
  saveRun,
  runPairedExperiment,
  fetchRunById,
  downloadRunExport,
  runOfflineDemo,
} from '../services/api'

function fmt(n) {
  if (n === null || n === undefined) return '—'
  return typeof n === 'number' ? n.toLocaleString() : String(n)
}

export default function Home({
  profile,
  onNewUpload,
  currentPhase = 1,
  onPhaseChange,
  currentRunId = null,
  onRunUpdated,
  initialRunState = null,
}) {
  // Local active phase state synchronized with parent
  const [activePhase, setActivePhase] = useState(currentPhase)

  // Experiment Run ID tracking
  const [activeRunId, setActiveRunId] = useState(currentRunId || initialRunState?.run_id || null)

  // Pipeline Data State (pre-populated if opening a saved run)
  const [objective, setObjective] = useState(initialRunState?.objective || 'Investigate this dataset.')
  const [plan, setPlan] = useState(initialRunState?.plan || null)
  const [analysisResponse, setAnalysisResponse] = useState(initialRunState?.analysis || null)
  const [criticResponse, setCriticResponse] = useState(initialRunState?.critic || null)
  const [verificationResponse, setVerificationResponse] = useState(initialRunState?.verification || null)
  const [insightsResponse, setInsightsResponse] = useState(initialRunState?.insights || null)
  const [evaluationResponse, setEvaluationResponse] = useState(initialRunState?.evaluation || null)

  // Stage C Offline Demonstration State
  const [isOfflineDemo, setIsOfflineDemo] = useState(Boolean(initialRunState?.is_offline_demo))
  const [runningDemo, setRunningDemo] = useState(false)

  // Persistence status state (Requirement 8)
  const [saveStatus, setSaveStatus] = useState('idle') // 'idle' | 'saving' | 'saved' | 'error'
  const [saveError, setSaveError] = useState('')
  const [exportingFormat, setExportingFormat] = useState(null)
  const [exportError, setExportError] = useState('')


  // Loading States
  const [planning, setPlanning] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [critiquing, setCritiquing] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [generatingInsights, setGeneratingInsights] = useState(false)
  const [evaluating, setEvaluating] = useState(false)

  // Error States
  const [planError, setPlanError] = useState('')
  const [analysisError, setAnalysisError] = useState('')
  const [criticError, setCriticError] = useState('')
  const [verificationError, setVerificationError] = useState('')
  const [insightsError, setInsightsError] = useState('')
  const [evaluationError, setEvaluationError] = useState('')

  // Experiment Benchmark Mode & Paired View State
  const [resultView, setResultView] = useState('proposed') // 'compare' | 'baseline' | 'proposed'
  const [pairedExperiment, setPairedExperiment] = useState(null)
  const [runningPaired, setRunningPaired] = useState(false)
  const [pairedError, setPairedError] = useState('')
  const [cachedBranchRuns, setCachedBranchRuns] = useState({ baseline: null, proposed: null })
  const [loadingBranchRun, setLoadingBranchRun] = useState(false)

  // Sync phase change with parent (for Navbar)
  const changePhase = (newPhase) => {
    setActivePhase(newPhase)
    if (onPhaseChange) {
      onPhaseChange(newPhase)
    }
  }

  // Calculate Max Unlocked Phase
  let maxUnlockedPhase = 1
  if (profile) maxUnlockedPhase = 2
  if (plan) maxUnlockedPhase = 3
  if (analysisResponse) maxUnlockedPhase = 4
  if (criticResponse) maxUnlockedPhase = 5
  if (verificationResponse) maxUnlockedPhase = 6
  if (insightsResponse) maxUnlockedPhase = 7

  // Calculate Completed Phases
  const completedPhases = []
  if (profile) completedPhases.push(1)
  if (plan) completedPhases.push(2)
  if (analysisResponse) completedPhases.push(3)
  if (criticResponse) completedPhases.push(4)
  if (verificationResponse) completedPhases.push(5)
  if (insightsResponse) completedPhases.push(6)
  if (evaluationResponse) completedPhases.push(7)

  // Helper to persist current run progress to backend/data/runs/ (Requirement 1, 8)
  const persistRun = async (overrides = {}) => {
    const payload = {
      run_id: overrides.run_id || activeRunId,
      dataset_name: profile?.filename || profile?.dataset_id || 'dataset.csv',
      dataset_id: profile?.dataset_id || profile?.filename || 'sample.csv',
      active_phase: overrides.active_phase ?? activePhase,
      objective: overrides.objective ?? objective,
      mode: overrides.mode !== undefined ? overrides.mode : (initialRunState?.mode || null),
      paired_run_id: overrides.paired_run_id !== undefined ? overrides.paired_run_id : (initialRunState?.paired_run_id || null),
      experiment_id: overrides.experiment_id !== undefined ? overrides.experiment_id : (initialRunState?.experiment_id || null),
      profile: profile,
      plan: overrides.plan !== undefined ? overrides.plan : plan,
      analysis: overrides.analysis !== undefined ? overrides.analysis : analysisResponse,
      critic: overrides.critic !== undefined ? overrides.critic : criticResponse,
      verification: overrides.verification !== undefined ? overrides.verification : verificationResponse,
      insights: overrides.insights !== undefined ? overrides.insights : insightsResponse,
      evaluation: overrides.evaluation !== undefined ? overrides.evaluation : evaluationResponse,
    }

    setSaveStatus('saving')
    setSaveError('')
    try {
      const res = await saveRun(payload)
      if (res?.run_id && res.run_id !== activeRunId) {
        setActiveRunId(res.run_id)
      }
      setSaveStatus('saved')
      if (onRunUpdated) {
        onRunUpdated(res.summary || res)
      }
      return res
    } catch (err) {
      setSaveStatus('error')
      setSaveError(err.message || 'Auto-save failed')
      console.error('Error auto-saving experiment run:', err)
    }
  }

  // Restore state when initialRunState changes (Requirement 5)
  useEffect(() => {
    if (initialRunState) {
      setActiveRunId(initialRunState.run_id || null)
      setObjective(initialRunState.objective || 'Investigate this dataset.')
      setPlan(initialRunState.plan || null)
      setAnalysisResponse(initialRunState.analysis || null)
      setCriticResponse(initialRunState.critic || null)
      setVerificationResponse(initialRunState.verification || null)
      setInsightsResponse(initialRunState.insights || null)
      setEvaluationResponse(initialRunState.evaluation || null)
      if (initialRunState.active_phase) {
        changePhase(initialRunState.active_phase)
      }
      setSaveStatus('saved')

      // Detect experiment mode if saved run belongs to baseline or proposed branch
      if (initialRunState.mode === 'baseline') {
        setResultView('baseline')
      } else if (initialRunState.mode === 'proposed') {
        setResultView('proposed')
      }
    }
  }, [initialRunState])

  // Auto-save initial profile on upload if not already saved
  useEffect(() => {
    if (profile && !initialRunState && !activeRunId) {
      persistRun({ active_phase: 1 })
    }
  }, [profile])

  // Handlers for Pipeline Actions
  const handleGeneratePlan = async (e, planOptions = {}) => {
    e?.preventDefault()
    if (!profile) return

    setPlanning(true)
    setPlanError('')
    setAnalysisResponse(null)
    setAnalysisError('')
    setCriticResponse(null)
    setCriticError('')
    setVerificationResponse(null)
    setVerificationError('')
    setInsightsResponse(null)
    setInsightsError('')

    try {
      const planRes = await createInvestigationPlan(profile, objective, planOptions)
      setPlan(planRes)
      await persistRun({ plan: planRes, active_phase: 2 })
    } catch (err) {
      setPlanError(err.message || 'Unable to create an investigation plan.')
    } finally {
      setPlanning(false)
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
    setInsightsResponse(null)
    setInsightsError('')

    try {
      const datasetId = profile?.dataset_id || profile?.filename || 'sample.csv'
      const response = await executeAnalysisPlan(datasetId, plan.analysis_tasks)
      setAnalysisResponse(response)
      await persistRun({ analysis: response, active_phase: 3 })
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
    setInsightsResponse(null)
    setInsightsError('')

    try {
      const response = await runCriticReview(analysisResponse, profile, plan)
      setCriticResponse(response)
      await persistRun({ critic: response, active_phase: 4 })
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
    setInsightsResponse(null)
    setInsightsError('')

    try {
      const response = await verifyFindings(analysisResponse, criticResponse, profile, plan)
      setVerificationResponse(response)
      await persistRun({ verification: response, active_phase: 5 })
    } catch (err) {
      setVerificationError(
        err.message || 'Failed to execute Phase 5 verification and re-investigation.'
      )
    } finally {
      setVerifying(false)
    }
  }

  const handleGenerateInsights = async () => {
    if (!verificationResponse) return

    setGeneratingInsights(true)
    setInsightsError('')

    try {
      const response = await generateInsights(verificationResponse, profile, plan)
      setInsightsResponse(response)
      await persistRun({ insights: response, active_phase: 6 })
    } catch (err) {
      setInsightsError(err.message || 'Failed to generate analytical insights.')
    } finally {
      setGeneratingInsights(false)
    }
  }

  const handleRunEvaluation = async () => {
    if (!profile) return

    setEvaluating(true)
    setEvaluationError('')

    try {
      const response = await evaluatePipeline({
        profile,
        plan,
        analysis: analysisResponse,
        critic: criticResponse,
        verification: verificationResponse,
        insights: insightsResponse,
      })
      setEvaluationResponse(response)
      await persistRun({ evaluation: response, active_phase: 7 })
    } catch (err) {
      setEvaluationError(err.message || 'Failed to evaluate system pipeline.')
    } finally {
      setEvaluating(false)
    }
  }

  // Branch Run Application (No Rerun)
  const applyBranchRun = (runData) => {
    if (!runData) return
    setActiveRunId(runData.run_id || null)
    setPlan(runData.plan || null)
    setAnalysisResponse(runData.analysis || null)
    setCriticResponse(runData.critic || null)
    setVerificationResponse(runData.verification || null)
    setInsightsResponse(runData.insights || null)
    setEvaluationResponse(runData.evaluation || null)
  }

  // Switching result views: purely loads in-memory or cached run; never reruns the experiment
  const handleChangeResultView = async (newView) => {
    if (newView === resultView) return
    setResultView(newView)

    if (newView === 'compare') {
      if (!pairedExperiment && (initialRunState?.paired_run_id || activeRunId)) {
        setLoadingBranchRun(true)
        setPairedError('')
        try {
          const pairedTargetId = initialRunState?.paired_run_id
          if (!pairedTargetId) {
            throw new Error('This saved run does not have a linked paired experiment reference.')
          }

          // Fetch the paired branch run
          const pairedRun = await fetchRunById(pairedTargetId)
          if (!pairedRun) {
            throw new Error(`Linked paired run '${pairedTargetId}' could not be retrieved.`)
          }

          // Determine which is baseline and which is proposed
          const currentMode = initialRunState?.mode || (initialRunState?.run_id?.includes('baseline') ? 'baseline' : 'proposed')
          const isCurrentBaseline = currentMode === 'baseline'
          const bRun = isCurrentBaseline ? (cachedBranchRuns.baseline || initialRunState) : pairedRun
          const pRun = isCurrentBaseline ? pairedRun : (cachedBranchRuns.proposed || initialRunState)

          // Validate that both runs contain evaluation data to support comparative benchmarking
          if (!bRun.evaluation?.summary || !pRun.evaluation?.summary) {
            const incompleteName = !bRun.evaluation?.summary ? bRun.run_id : pRun.run_id
            throw new Error(`Saved comparison data cannot be displayed: paired run '${incompleteName}' is incomplete or missing evaluation records.`)
          }

          // Construct authoritative comparison object from real persisted experiment data
          const bSummary = bRun.evaluation.summary
          const pSummary = pRun.evaluation.summary
          const bScore = Number(bSummary.overall_score ?? 0)
          const pScore = Number(pSummary.overall_score ?? 0)
          const scoreDelta = pScore - bScore

          // Extract dimension deltas
          const bDims = bRun.evaluation.dimensions || {}
          const pDims = pRun.evaluation.dimensions || {}
          const dimDeltas = {}
          const allDimKeys = new Set([...Object.keys(bDims), ...Object.keys(pDims)])
          for (const k of allDimKeys) {
            const bVal = Number(bDims[k]?.score ?? bDims[k] ?? 0)
            const pVal = Number(pDims[k]?.score ?? pDims[k] ?? 0)
            dimDeltas[k] = pVal - bVal
          }

          const violationsFlagged = pRun.critic?.summary?.requires_reinvestigation ?? pRun.critic?.summary?.flagged ?? 0
          const violationsResolved = pRun.verification?.summary?.reinvestigations_executed ?? pRun.verification?.summary?.verified ?? 0

          const reconstructedExperiment = {
            experiment_id: bRun.experiment_id || pRun.experiment_id || `exp_restored_${bRun.run_id.slice(-6)}`,
            dataset_id: bRun.dataset_id || pRun.dataset_id || profile?.filename || 'dataset.csv',
            objective: bRun.objective || pRun.objective || objective || 'Objective not available',
            created_at: pRun.created_at || bRun.created_at,
            persisted: true,
            shared_plan: bRun.plan || pRun.plan || null,
            baseline: {
              run_id: bRun.run_id,
              mode: 'baseline',
              dataset_id: bRun.dataset_id,
              total_tasks_executed: bRun.analysis?.successful_tasks ?? bSummary.total_tasks_executed ?? 0,
              overall_score: bScore,
              overall_grade: bSummary.overall_grade || '—',
              dimension_scores: Object.fromEntries(Object.entries(bDims).map(([k, v]) => [k, typeof v === 'object' ? v.score : v])),
              completed_phases: bRun.completed_phases || [1, 2, 3, 7],
              has_critic_review: false,
              has_verification: false,
              has_insights: false,
              persisted: true,
              paired_run_id: pRun.run_id,
            },
            proposed: {
              run_id: pRun.run_id,
              mode: 'proposed',
              dataset_id: pRun.dataset_id,
              total_tasks_executed: pRun.analysis?.successful_tasks ?? pSummary.total_tasks_executed ?? 0,
              overall_score: pScore,
              overall_grade: pSummary.overall_grade || '—',
              dimension_scores: Object.fromEntries(Object.entries(pDims).map(([k, v]) => [k, typeof v === 'object' ? v.score : v])),
              completed_phases: pRun.completed_phases || [1, 2, 3, 4, 5, 6, 7],
              has_critic_review: Boolean(pRun.critic),
              has_verification: Boolean(pRun.verification),
              has_insights: Boolean(pRun.insights),
              persisted: true,
              paired_run_id: bRun.run_id,
            },
            comparison: {
              baseline_score: bScore,
              proposed_score: pScore,
              overall_score_delta: scoreDelta,
              rigor_score_delta: dimDeltas.statistical_rigor ?? 0,
              verification_score_delta: dimDeltas.truth_grounding ?? 0,
              dimension_deltas: dimDeltas,
              assumption_violations_flagged: violationsFlagged,
              violations_resolved_via_reinvestigation: violationsResolved,
              resolution_rate: violationsFlagged > 0 ? (violationsResolved / violationsFlagged) : 1.0,
              total_reinvestigations: violationsResolved,
              verified_findings_count: pRun.verification?.summary?.verified ?? 0,
              questionable_findings_count: pRun.critic?.summary?.questionable ?? 0,
              rejected_findings_count: pRun.critic?.summary?.rejected ?? 0,
              initial_findings_count: pRun.analysis?.successful_tasks ?? bRun.analysis?.successful_tasks ?? 0,
              findings_accepted_directly: pRun.verification?.summary?.verified ?? 0,
              findings_requiring_reinvestigation: violationsFlagged,
              findings_revised: violationsResolved,
              findings_unresolved: (pRun.critic?.summary?.questionable || 0) + (pRun.critic?.summary?.rejected || 0),
              scientific_verdict: scoreDelta > 0
                ? `Proposed pipeline improved overall score by +${scoreDelta.toFixed(1)} pts (${bScore.toFixed(1)} -> ${pScore.toFixed(1)}) with empirical verification.`
                : scoreDelta < 0
                ? `Proposed pipeline score was lower by ${Math.abs(scoreDelta).toFixed(1)} pts (${bScore.toFixed(1)} -> ${pScore.toFixed(1)}) due to strict criticism of unverified findings.`
                : `Proposed and baseline pipelines achieved equal overall score (${bScore.toFixed(1)} pts).`,
            },
          }

          setCachedBranchRuns({ baseline: bRun, proposed: pRun })
          setPairedExperiment(reconstructedExperiment)
        } catch (err) {
          setPairedError(err.message || 'Failed to restore historical comparison data.')
          console.error('Error hydrating historical paired experiment:', err)
        } finally {
          setLoadingBranchRun(false)
        }
      }
    } else if (newView === 'baseline') {
      if (cachedBranchRuns.baseline) {
        applyBranchRun(cachedBranchRuns.baseline)
      } else if (pairedExperiment?.baseline?.run_id) {
        setLoadingBranchRun(true)
        try {
          const bRun = await fetchRunById(pairedExperiment.baseline.run_id)
          setCachedBranchRuns((prev) => ({ ...prev, baseline: bRun }))
          applyBranchRun(bRun)
        } catch (e) {
          console.error('Failed to load baseline branch details:', e)
        } finally {
          setLoadingBranchRun(false)
        }
      } else if (initialRunState?.mode === 'proposed' && initialRunState?.paired_run_id) {
        setLoadingBranchRun(true)
        try {
          const bRun = await fetchRunById(initialRunState.paired_run_id)
          setCachedBranchRuns((prev) => ({ ...prev, baseline: bRun }))
          applyBranchRun(bRun)
        } catch (e) {
          console.error('Failed to load baseline branch details:', e)
        } finally {
          setLoadingBranchRun(false)
        }
      }
    } else if (newView === 'proposed') {
      if (cachedBranchRuns.proposed) {
        applyBranchRun(cachedBranchRuns.proposed)
      } else if (pairedExperiment?.proposed?.run_id) {
        setLoadingBranchRun(true)
        try {
          const pRun = await fetchRunById(pairedExperiment.proposed.run_id)
          setCachedBranchRuns((prev) => ({ ...prev, proposed: pRun }))
          applyBranchRun(pRun)
        } catch (e) {
          console.error('Failed to load proposed branch details:', e)
        } finally {
          setLoadingBranchRun(false)
        }
      } else if (initialRunState?.mode === 'baseline' && initialRunState?.paired_run_id) {
        setLoadingBranchRun(true)
        try {
          const pRun = await fetchRunById(initialRunState.paired_run_id)
          setCachedBranchRuns((prev) => ({ ...prev, proposed: pRun }))
          applyBranchRun(pRun)
        } catch (e) {
          console.error('Failed to load proposed branch details:', e)
        } finally {
          setLoadingBranchRun(false)
        }
      }
    }
  }

  // Export current active run
  const handleExportRun = async (format) => {
    if (!activeRunId) return
    setExportingFormat(format)
    setExportError('')
    try {
      await downloadRunExport(activeRunId, format)
    } catch (err) {
      setExportError(err.message || `Failed to export ${format}`)
      console.error('Export error:', err)
    } finally {
      setExportingFormat(null)
    }
  }

  // Handler for Comparative Experiment Action
  const handleRunPairedExperiment = async () => {
    if (!profile) return

    setRunningPaired(true)
    setPairedError('')

    try {
      const datasetId = profile?.dataset_id || profile?.filename || 'sample.csv'
      const response = await runPairedExperiment({
        datasetId,
        profile,
        objective,
        maxReinvestigations: 1,
        persistRuns: true,
      })

      setPairedExperiment(response)
      setResultView('compare')

      // Pre-fetch both branch runs for instant, zero-latency view toggling
      if (response.baseline?.run_id && response.proposed?.run_id) {
        try {
          const [bRun, pRun] = await Promise.all([
            fetchRunById(response.baseline.run_id),
            fetchRunById(response.proposed.run_id),
          ])
          setCachedBranchRuns({ baseline: bRun, proposed: pRun })
        } catch (cacheErr) {
          console.warn('Could not pre-fetch full branch details from repository:', cacheErr)
        }
      }

      if (onRunUpdated) {
        onRunUpdated(response.proposed)
      }
    } catch (err) {
      setPairedError(err.message || 'Failed to execute comparative experiment.')
    } finally {
      setRunningPaired(false)
    }
  }

  const activePhaseInfo = getPhaseById(activePhase)
  const { overview, missing, duplicates } = profile || {}
  const missingVariant =
    missing?.total_missing === 0
      ? 'success'
      : missing?.total_missing_pct < 5
      ? 'warning'
      : 'error'
  const dupVariant = duplicates?.n_duplicates === 0 ? 'success' : 'warning'

  const handleRunOfflineDemo = async () => {
    try {
      setRunningDemo(true)
      const demoData = await runOfflineDemo()
      setObjective(demoData.objective)
      setPlan(demoData.plan)
      setAnalysisResponse(demoData.analysis)
      setCriticResponse(demoData.critic)
      setVerificationResponse(demoData.verification)
      setInsightsResponse(demoData.insights)
      setEvaluationResponse(demoData.evaluation)
      setIsOfflineDemo(true)
      setResultView('single')
      changePhase(7)
    } catch (err) {
      console.error('Failed to run offline demo:', err)
      alert(err.message || 'Offline demo execution failed.')
    } finally {
      setRunningDemo(false)
    }
  }

  return (
    <div className="home fade-in">
      {/* ── Top Stepper: Investigation Journey Navigation ───────── */}
      <InvestigationStepper
        phases={PHASES}
        currentPhase={activePhase}
        maxUnlockedPhase={maxUnlockedPhase}
        completedPhases={completedPhases}
        onSelectPhase={(phaseId) => changePhase(phaseId)}
      />

      {/* ── Active Phase Stage Header ──────────────────────────── */}
      <div className="phase-stage-header">
        <div className="phase-stage-title-wrap">
          <span className="phase-stage-title">{activePhaseInfo.title}</span>
          <span className="badge badge-cyan">{activePhaseInfo.badge}</span>
          {saveStatus === 'saving' && (
            <span className="badge badge-cyan" style={{ fontSize: '0.75rem', opacity: 0.85 }}>
              💾 Saving...
            </span>
          )}
          {saveStatus === 'saved' && (
            <span className="badge badge-green" style={{ fontSize: '0.75rem' }}>
              ✓ Saved
            </span>
          )}
          {saveStatus === 'error' && (
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <span className="badge" style={{ background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', fontSize: '0.75rem' }}>
                ⚠ Save Failed: {saveError}
              </span>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                style={{ fontSize: '0.7rem', padding: '2px 8px' }}
                onClick={() => persistRun()}
              >
                Retry
              </button>
            </div>
          )}
        </div>
        <div style={{ display: 'inline-flex', gap: '8px', alignItems: 'center' }}>
          {activeRunId && (
            <>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={() => handleExportRun('markdown')}
                disabled={exportingFormat === 'markdown'}
                title="Download complete Markdown research report for this run"
              >
                {exportingFormat === 'markdown' ? '...' : '📥 Export .MD'}
              </button>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={() => handleExportRun('json')}
                disabled={exportingFormat === 'json'}
                title="Download structured JSON run state & report"
              >
                {exportingFormat === 'json' ? '...' : '{ } Export JSON'}
              </button>
            </>
          )}
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={handleRunOfflineDemo}
            disabled={runningDemo}
            title="Execute 100% offline demonstration using synthetic data without LLM API calls"
            style={{ borderColor: 'rgba(56, 189, 248, 0.4)', color: '#38bdf8' }}
          >
            {runningDemo ? 'Running Demo...' : '⚡ Offline Demo'}
          </button>
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={onNewUpload}
            title="Reset pipeline and upload a new dataset"
          >
            ↑ New Dataset
          </button>
        </div>
      </div>

      {/* ── Export Error Alert Banner ──────────────────────────────── */}
      {exportError && (
        <div
          className="export-error-banner fade-in"
          role="alert"
          style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            color: '#fca5a5',
            padding: '10px 16px',
            borderRadius: '8px',
            marginBottom: '16px',
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
            className="btn btn-outline btn-sm"
            style={{ fontSize: '0.75rem', padding: '2px 8px', borderColor: 'rgba(239, 68, 68, 0.4)', color: '#fca5a5' }}
            onClick={() => setExportError('')}
            aria-label="Dismiss export error"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ── Benchmark Mode Switcher (Compare / Baseline / Proposed) ─ */}
      {isOfflineDemo && (
        <div
          className="card card-no-hover fade-in"
          style={{
            background: 'rgba(56, 189, 248, 0.08)',
            border: '1px solid rgba(56, 189, 248, 0.35)',
            padding: '12px 18px',
            marginBottom: '16px',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <span style={{ fontSize: '1.3rem' }}>⚡</span>
          <div style={{ fontSize: '0.825rem', color: '#e2e8f0', lineHeight: 1.45 }}>
            <strong style={{ color: '#38bdf8' }}>OFFLINE DEMONSTRATION MODE:</strong> All outputs below are simulated and deterministic using synthetic test data. Zero Gemini LLM API calls were made. Research experiments strictly enforce genuine Gemini LLM planning with zero deterministic fallback.
          </div>
        </div>
      )}

      {(pairedExperiment || initialRunState?.mode || initialRunState?.paired_run_id) && (
        <ExperimentModeToggle
          activeMode={resultView}
          onChangeMode={handleChangeResultView}
          baselineScore={
            pairedExperiment?.baseline?.overall_score ??
            (initialRunState?.mode === 'baseline' ? initialRunState?.evaluation?.summary?.overall_score : null)
          }
          proposedScore={
            pairedExperiment?.proposed?.overall_score ??
            (initialRunState?.mode === 'proposed' ? initialRunState?.evaluation?.summary?.overall_score : null)
          }
          overallDelta={pairedExperiment?.comparison?.overall_score_delta}
          baselineGrade={
            pairedExperiment?.baseline?.overall_grade ??
            (initialRunState?.mode === 'baseline' ? initialRunState?.evaluation?.summary?.overall_grade : null)
          }
          proposedGrade={
            pairedExperiment?.proposed?.overall_grade ??
            (initialRunState?.mode === 'proposed' ? initialRunState?.evaluation?.summary?.overall_grade : null)
          }
        />
      )}

      {/* ── Paired Experiment Loading Progress State ─────────────── */}
      {runningPaired && (
        <div
          className="card card-no-hover fade-in"
          style={{
            background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95))',
            border: '1px solid rgba(99, 102, 241, 0.4)',
            padding: '24px',
            borderRadius: '12px',
            marginBottom: '24px',
            textAlign: 'center',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          }}
        >
          <div
            className="planner-spinner"
            style={{ width: '32px', height: '32px', margin: '0 auto 14px auto', borderWidth: '3px' }}
          ></div>
          <h3 style={{ color: '#f8fafc', fontSize: '1.2rem', marginBottom: '8px' }}>
            Executing Paired Experiment Benchmark (Baseline vs Proposed)...
          </h3>
          <p
            style={{
              color: '#94a3b8',
              fontSize: '0.88rem',
              maxWidth: '620px',
              margin: '0 auto 14px auto',
              lineHeight: 1.5,
            }}
          >
            Generating single authoritative plan, computing identical SciPy statistical analyses, executing baseline and proposed pipelines, and calculating comparative delta metrics.
          </p>
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '8px',
              fontSize: '0.78rem',
              color: '#a5b4fc',
              flexWrap: 'wrap',
            }}
          >
            <span>1. Authoritative Plan</span> <span>→</span>
            <span>2. SciPy Analysis</span> <span>→</span>
            <span>3. Baseline Eval</span> <span>→</span>
            <span>4. Proposed Critic & Re-test</span> <span>→</span>
            <span>5. Comparison Scorecard</span>
          </div>
        </div>
      )}

      {/* ── Paired Experiment Error Banner ────────────────────────── */}
      {pairedError && (
        <div
          className="planner-warnings fade-in"
          style={{
            borderColor: 'var(--error)',
            background: 'rgba(239, 68, 68, 0.08)',
            width: '100%',
            marginBottom: '24px',
            padding: '16px',
            borderRadius: '8px',
          }}
        >
          <div
            className="warnings-header"
            style={{ color: 'var(--error)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}
          >
            <span>⚠</span>
            <strong>Comparative Experiment Error</strong>
          </div>
          <div className="warning-item" style={{ color: 'var(--text-primary)', fontSize: '0.88rem' }}>
            {pairedError}
          </div>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleRunPairedExperiment}
            style={{ marginTop: '10px', padding: '6px 14px', fontSize: '0.82rem' }}
          >
            Retry Comparative Experiment
          </button>
        </div>
      )}

      {/* ── Loading Branch Details Indicator (Instant Cache Switching) ── */}
      {loadingBranchRun && (
        <div
          className="fade-in"
          style={{ textAlign: 'center', padding: '12px', color: '#818cf8', fontSize: '0.85rem' }}
        >
          <span>Loading branch details from run store...</span>
        </div>
      )}

      {/* ── VIEW 1: COMPARE BOTH VIEW ────────────────────────────── */}
      {resultView === 'compare' && pairedExperiment ? (
        <CompareExperimentsView
          experiment={pairedExperiment}
          onViewBaseline={() => handleChangeResultView('baseline')}
          onViewProposed={() => handleChangeResultView('proposed')}
        />
      ) : (
        <>
          {/* ── Compact Completed Phase Summaries ──────────────────── */}
          <CompletedPhaseSummary
            currentPhase={activePhase}
            profile={profile}
            plan={plan}
            analysisResponse={analysisResponse}
            criticResponse={criticResponse}
            verificationResponse={verificationResponse}
            insightsResponse={insightsResponse}
            evaluationResponse={evaluationResponse}
            onNavigate={(phaseId) => changePhase(phaseId)}
          />

      {/* ── PHASE 1: DATASET PROFILE ───────────────────────────── */}
      {activePhase === 1 && (
        <div className="phase-content fade-in">
          {/* Dataset Metric Cards Row */}
          <div className="metrics-grid">
            <MetricCard label="Total Rows" value={fmt(overview?.n_rows)} icon="⬛" />
            <MetricCard label="Columns" value={fmt(overview?.n_cols)} icon="⫿" />
            <MetricCard
              label="Missing Values"
              value={fmt(missing?.total_missing)}
              sub={`${missing?.total_missing_pct}% of cells`}
              icon="△"
              variant={missingVariant}
            />
            <MetricCard
              label="Duplicate Rows"
              value={fmt(duplicates?.n_duplicates)}
              sub={`${duplicates?.duplicate_pct}% of rows`}
              icon="⧉"
              variant={dupVariant}
            />
            <MetricCard
              label="Memory"
              value={`${((overview?.memory_usage_bytes || 0) / 1024).toFixed(1)} KB`}
              icon="◈"
            />
          </div>

          {/* Two-Column Profiling Dashboard Layout */}
          <div className="dashboard-layout">
            <div className="dash-left">
              <div className="card card-no-hover section-card">
                <DataQuality profile={profile} />
              </div>
              <div className="card card-no-hover section-card">
                <NumericalProfile profile={profile} />
              </div>
            </div>
            <div className="dash-right">
              <div className="card card-no-hover section-card">
                <ColumnProfile profile={profile} />
              </div>
            </div>
          </div>

          {/* Dataset Preview */}
          <div className="card card-no-hover section-card">
            <DatasetPreview profile={profile} />
          </div>

          {/* Bottom Call-to-Action to Proceed to Phase 2 */}
          <div className="phase-transition-cta card card-no-hover fade-in">
            <div className="cta-content">
              <div className="cta-title">Dataset Profiling Complete</div>
              <div className="cta-subtitle">
                Review data quality and column metrics above, then formulate your analytical
                objective to generate structured investigation tasks.
              </div>
            </div>
            <button
              type="button"
              className="btn btn-primary phase-action-btn"
              onClick={() => changePhase(2)}
            >
              <span>Continue to Investigation Plan (Phase 2)</span>
              <span>→</span>
            </button>
          </div>
        </div>
      )}

      {/* ── PHASE 2: INVESTIGATION PLANNER ─────────────────────── */}
      {activePhase === 2 && (
        <div className="phase-content fade-in">
          <PlanSection
            profile={profile}
            plan={plan}
            objective={objective}
            setObjective={setObjective}
            loading={planning}
            error={planError}
            onGeneratePlan={handleGeneratePlan}
            onProceedToAnalyze={() => changePhase(3)}
            onRunComparativeExperiment={handleRunPairedExperiment}
            runningComparative={runningPaired}
          />
        </div>
      )}

      {/* ── PHASE 3: STATISTICAL ANALYSIS ──────────────────────── */}
      {activePhase === 3 && (
        <div className="phase-content fade-in">
          {!analysisResponse ? (
            <div className="phase-ready-card fade-in">
              <div className="phase-ready-icon">🔬</div>
              <div className="phase-ready-title">Ready to Execute Statistical Analysis</div>
              <div className="phase-ready-desc">
                Execute {plan?.analysis_tasks?.length || 0} planned statistical tasks (bivariate
                correlations, 2-group comparisons, ANOVA, contingency tables & outlier
                sensitivity) with exact SciPy calculations.
              </div>

              {analysisError && (
                <div
                  className="planner-warnings"
                  style={{
                    borderColor: 'var(--error)',
                    background: 'rgba(239, 68, 68, 0.08)',
                    width: '100%',
                    maxWidth: '600px',
                    textAlign: 'left',
                  }}
                >
                  <div className="warnings-header" style={{ color: 'var(--error)' }}>
                    <span>⚠</span>
                    <span>Analysis Error</span>
                  </div>
                  <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
                    {analysisError}
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap', marginTop: '16px' }}>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={handleRunAnalysis}
                  disabled={analyzing || !plan}
                >
                  {analyzing ? (
                    <>
                      <div className="planner-spinner"></div>
                      <span>Computing Statistical Tests (SciPy)...</span>
                    </>
                  ) : (
                    <>
                      <span>🔬</span>
                      <span>Execute Statistical Analysis (Phase 3)</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="btn btn-secondary phase-action-btn"
                  onClick={handleRunPairedExperiment}
                  disabled={runningPaired || !plan}
                  style={{ borderColor: 'rgba(99, 102, 241, 0.5)', background: 'rgba(99, 102, 241, 0.12)' }}
                  title="Run paired benchmark comparing Baseline vs Proposed"
                >
                  {runningPaired ? (
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
              </div>
            </div>
          ) : (
            <div className="fade-in">
              <AnalysisResults analysisResponse={analysisResponse} />

              {/* Bottom Call-to-Action to proceed to Phase 4 */}
              <div className="phase-transition-cta card card-no-hover fade-in">
                <div className="cta-content">
                  <div className="cta-title">Statistical Execution Complete</div>
                  <div className="cta-subtitle">
                    {analysisResponse.successful_tasks} statistical results calculated. Run the
                    Critic Agent to evaluate sample size adequacy, outlier distortion, and
                    methodological rigor.
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => changePhase(4)}
                >
                  <span>Proceed to Critic Verification (Phase 4)</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── PHASE 4: CRITIC AGENT VERIFICATION ─────────────────── */}
      {activePhase === 4 && (
        <div className="phase-content fade-in">
          {resultView === 'baseline' ? (
            <div className="phase-ready-card fade-in" style={{ borderColor: 'rgba(59, 130, 246, 0.4)' }}>
              <div className="phase-ready-icon">🔬</div>
              <div className="phase-ready-title">Critic Review Bypassed in Baseline Architecture</div>
              <div className="phase-ready-desc">
                In Baseline mode, raw statistical analysis results proceed directly to Phase 7 System Evaluation without Critic assumption verification or outlier audits.
              </div>
              <div style={{ display: 'flex', gap: '10px', marginTop: '16px', flexWrap: 'wrap', justifyContent: 'center' }}>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => changePhase(7)}
                >
                  <span>Proceed to Baseline Evaluation (Phase 7) →</span>
                </button>
                {pairedExperiment && (
                  <button
                    type="button"
                    className="btn btn-secondary phase-action-btn"
                    onClick={() => handleChangeResultView('proposed')}
                  >
                    <span>Switch to Proposed Architecture Mode</span>
                  </button>
                )}
              </div>
            </div>
          ) : !criticResponse ? (
            <div className="phase-ready-card fade-in">
              <div className="phase-ready-icon">🛡</div>
              <div className="phase-ready-title">Ready for Critic Verification Review</div>
              <div className="phase-ready-desc">
                Evaluate empirical findings for sample size adequacy, outlier distortion,
                distributional stability, and effect size relevance. Findings will be classified
                into Accepted, Questionable, Rejected, or Requires Re-investigation.
              </div>

              {criticError && (
                <div
                  className="planner-warnings"
                  style={{
                    borderColor: 'var(--error)',
                    background: 'rgba(239, 68, 68, 0.08)',
                    width: '100%',
                    maxWidth: '600px',
                    textAlign: 'left',
                  }}
                >
                  <div className="warnings-header" style={{ color: 'var(--error)' }}>
                    <span>⚠</span>
                    <span>Critic Error</span>
                  </div>
                  <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
                    {criticError}
                  </div>
                </div>
              )}

              <button
                type="button"
                className="btn btn-primary phase-action-btn"
                onClick={handleRunCritic}
                disabled={critiquing || !analysisResponse}
              >
                {critiquing ? (
                  <>
                    <div className="planner-spinner"></div>
                    <span>Evaluating Findings (Critic Agent)...</span>
                  </>
                ) : (
                  <>
                    <span>🛡</span>
                    <span>Run Critic Verification Review (Phase 4)</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="fade-in">
              <CriticSection criticResponse={criticResponse} />

              {/* Bottom Call-to-Action to proceed to Phase 5 */}
              <div className="phase-transition-cta card card-no-hover fade-in">
                <div className="cta-content">
                  <div className="cta-title">Critic Review Complete</div>
                  <div className="cta-subtitle">
                    {criticResponse.summary?.accepted || 0} findings verified,{' '}
                    {criticResponse.summary?.requires_reinvestigation || 0} require targeted
                    re-investigation. Proceed to final synthesis.
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => changePhase(5)}
                >
                  <span>Proceed to Finding Verification (Phase 5)</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── PHASE 5: VERIFIED FINDINGS & RE-INVESTIGATION ───────── */}
      {activePhase === 5 && (
        <div className="phase-content fade-in">
          {resultView === 'baseline' ? (
            <div className="phase-ready-card fade-in" style={{ borderColor: 'rgba(59, 130, 246, 0.4)' }}>
              <div className="phase-ready-icon">🔬</div>
              <div className="phase-ready-title">Re-investigation Bypassed in Baseline Architecture</div>
              <div className="phase-ready-desc">
                Controlled non-parametric alternative testing and finding verification are omitted in Baseline Mode to measure raw pipeline output.
              </div>
              <div style={{ display: 'flex', gap: '10px', marginTop: '16px', flexWrap: 'wrap', justifyContent: 'center' }}>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => changePhase(7)}
                >
                  <span>Proceed to Baseline Evaluation (Phase 7) →</span>
                </button>
                {pairedExperiment && (
                  <button
                    type="button"
                    className="btn btn-secondary phase-action-btn"
                    onClick={() => handleChangeResultView('proposed')}
                  >
                    <span>Switch to Proposed Architecture Mode</span>
                  </button>
                )}
              </div>
            </div>
          ) : !verificationResponse ? (
            <div className="phase-ready-card fade-in">
              <div className="phase-ready-icon">✨</div>
              <div className="phase-ready-title">
                Ready to Synthesize Verified Findings & Re-investigate
              </div>
              <div className="phase-ready-desc">
                Execute targeted non-parametric alternative tests for flagged hypotheses, compare
                initial vs re-tested statistical results, and synthesize final verified findings
                with full audit trail traceability.
              </div>

              {verificationError && (
                <div
                  className="planner-warnings"
                  style={{
                    borderColor: 'var(--error)',
                    background: 'rgba(239, 68, 68, 0.08)',
                    width: '100%',
                    maxWidth: '600px',
                    textAlign: 'left',
                  }}
                >
                  <div className="warnings-header" style={{ color: 'var(--error)' }}>
                    <span>⚠</span>
                    <span>Verification Error</span>
                  </div>
                  <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
                    {verificationError}
                  </div>
                </div>
              )}

              <button
                type="button"
                className="btn btn-primary phase-action-btn"
                onClick={handleVerifyFindings}
                disabled={verifying || !criticResponse}
              >
                {verifying ? (
                  <>
                    <div className="planner-spinner"></div>
                    <span>Re-investigating & Verifying Findings...</span>
                  </>
                ) : (
                  <>
                    <span>✨</span>
                    <span>Run Re-investigation & Synthesize Verified Findings (Phase 5)</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="fade-in">
              <VerifiedFindingsSection verificationResponse={verificationResponse} />

              {/* Bottom Call-to-Action to proceed to Phase 6 */}
              <div className="phase-transition-cta card card-no-hover fade-in">
                <div className="cta-content">
                  <div className="cta-title">Finding Verification Complete</div>
                  <div className="cta-subtitle">
                    {verificationResponse.summary?.verified || 0} findings verified and validated.
                    Proceed to generate clear, human-readable analytical insights.
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => changePhase(6)}
                >
                  <span>Proceed to Key Insights (Phase 6)</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── PHASE 6: KEY INSIGHTS ──────────────────────────────── */}
      {activePhase === 6 && (
        <div className="phase-content fade-in">
          {resultView === 'baseline' ? (
            <div className="phase-ready-card fade-in" style={{ borderColor: 'rgba(59, 130, 246, 0.4)' }}>
              <div className="phase-ready-icon">🔬</div>
              <div className="phase-ready-title">Insight Synthesis Bypassed in Baseline Architecture</div>
              <div className="phase-ready-desc">
                Grounded analytical insights generation is omitted in Baseline Mode. Raw statistical findings flow directly to Phase 7 evaluation.
              </div>
              <div style={{ display: 'flex', gap: '10px', marginTop: '16px', flexWrap: 'wrap', justifyContent: 'center' }}>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => changePhase(7)}
                >
                  <span>Proceed to Baseline Evaluation (Phase 7) →</span>
                </button>
                {pairedExperiment && (
                  <button
                    type="button"
                    className="btn btn-secondary phase-action-btn"
                    onClick={() => handleChangeResultView('proposed')}
                  >
                    <span>Switch to Proposed Architecture Mode</span>
                  </button>
                )}
              </div>
            </div>
          ) : !insightsResponse ? (
            <div className="phase-ready-card fade-in">
              <div className="phase-ready-icon">💡</div>
              <div className="phase-ready-title">Ready to Generate Key Analytical Insights</div>
              <div className="phase-ready-desc">
                Translate verified statistical findings into clear, human-readable, non-causal
                insights with transparent strength classification, full empirical evidence
                traceability, and documented limitations.
              </div>

              {insightsError && (
                <div
                  className="planner-warnings"
                  style={{
                    borderColor: 'var(--error)',
                    background: 'rgba(239, 68, 68, 0.08)',
                    width: '100%',
                    maxWidth: '600px',
                    textAlign: 'left',
                  }}
                >
                  <div className="warnings-header" style={{ color: 'var(--error)' }}>
                    <span>⚠</span>
                    <span>Insight Generation Error</span>
                  </div>
                  <div className="warning-item" style={{ color: 'var(--text-primary)' }}>
                    {insightsError}
                  </div>
                </div>
              )}

              <button
                type="button"
                className="btn btn-primary phase-action-btn"
                onClick={handleGenerateInsights}
                disabled={generatingInsights || !verificationResponse}
              >
                {generatingInsights ? (
                  <>
                    <div className="planner-spinner"></div>
                    <span>Synthesizing Key Insights...</span>
                  </>
                ) : (
                  <>
                    <span>💡</span>
                    <span>Generate Key Insights (Phase 6)</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="fade-in">
              <InsightsSection insightsResponse={insightsResponse} />

              {/* Bottom Call-to-Action to proceed to Phase 7 */}
              <div className="phase-transition-cta card card-no-hover fade-in">
                <div className="cta-content">
                  <div className="cta-title">Key Insights Synthesized</div>
                  <div className="cta-subtitle">
                    {insightsResponse.summary?.total_insights || 0} insights generated with full empirical traceability.
                    Proceed to System Evaluation and AutoMind Benchmark Scorecard.
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-primary phase-action-btn"
                  onClick={() => {
                    changePhase(7)
                    if (!evaluationResponse) {
                      handleRunEvaluation()
                    }
                  }}
                >
                  <span>Proceed to System Evaluation (Phase 7)</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── PHASE 7: SYSTEM EVALUATION & BENCHMARK SCORECARD ────── */}
      {activePhase === 7 && (
        <div className="phase-content fade-in">
          <EvaluationSection
            evaluationResponse={evaluationResponse}
            evaluating={evaluating}
            onRunEvaluation={handleRunEvaluation}
            evaluationError={evaluationError}
            activeRunId={activeRunId}
            onExportRun={handleExportRun}
            exportingFormat={exportingFormat}
          />
        </div>
      )}
        </>
      )}
    </div>
  )
}

