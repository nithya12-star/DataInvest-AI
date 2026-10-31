// components/CompletedPhaseSummary.jsx — Compact previous completed phase summary chips
import './CompletedPhaseSummary.css'

export default function CompletedPhaseSummary({
  currentPhase,
  profile,
  plan,
  analysisResponse,
  criticResponse,
  verificationResponse,
  insightsResponse,
  evaluationResponse,
  onNavigate,
}) {

  const items = []

  // Phase 1 summary if completed and we are currently viewing another phase
  if (profile && currentPhase !== 1) {
    items.push({
      phaseId: 1,
      name: 'Profile',
      meta: `${profile.overview?.n_rows?.toLocaleString() || 0} rows · ${
        profile.overview?.n_cols || 0
      } columns · ${profile.missing?.total_missing || 0} missing`,
      btnText: 'View Profile',
    })
  }

  // Phase 2 summary if completed and we are currently viewing another phase
  if (plan && currentPhase !== 2) {
    items.push({
      phaseId: 2,
      name: 'Plan',
      meta: `${plan.analysis_tasks?.length || 0} investigation tasks planned`,
      btnText: 'View Plan',
    })
  }

  // Phase 3 summary if completed and we are currently viewing another phase
  if (analysisResponse && currentPhase !== 3) {
    items.push({
      phaseId: 3,
      name: 'Analyze',
      meta: `${analysisResponse.successful_tasks || 0} statistical tests executed via SciPy`,
      btnText: 'View Analysis',
    })
  }

  // Phase 4 summary if completed and we are currently viewing another phase
  if (criticResponse && currentPhase !== 4) {
    items.push({
      phaseId: 4,
      name: 'Critic',
      meta: `${criticResponse.summary?.accepted || 0} Accepted · ${
        criticResponse.summary?.questionable || 0
      } Questionable · ${criticResponse.summary?.requires_reinvestigation || 0} Re-investigate`,
      btnText: 'View Critic',
    })
  }

  // Phase 5 summary if completed and we are currently viewing another phase
  if (verificationResponse && currentPhase !== 5) {
    items.push({
      phaseId: 5,
      name: 'Verify',
      meta: `${verificationResponse.summary?.verified || 0} Verified findings · Integrity: ${
        verificationResponse.summary?.overall_integrity || 'MODERATE'
      }`,
      btnText: 'View Verified',
    })
  }

  // Phase 6 summary if completed and we are currently viewing another phase
  if (insightsResponse && currentPhase !== 6) {
    items.push({
      phaseId: 6,
      name: 'Insights',
      meta: `${insightsResponse.summary?.total_insights || 0} Key analytical insights generated`,
      btnText: 'View Insights',
    })
  }

  // Phase 7 summary if completed and we are currently viewing another phase
  if (evaluationResponse && currentPhase !== 7) {
    items.push({
      phaseId: 7,
      name: 'Evaluation',
      meta: `Score: ${evaluationResponse.summary?.overall_score || 0}/100 (${
        evaluationResponse.summary?.overall_grade || 'A'
      })`,
      btnText: 'View Evaluation',
    })
  }

  if (items.length === 0) return null


  return (
    <div className="pipeline-history-container fade-in">
      {items.map((item) => (
        <div key={item.phaseId} className="history-strip">
          <div className="history-left">
            <div className="history-check">✓</div>
            <div>
              <span className="history-title">{item.name}: </span>
              <span className="history-meta">{item.meta}</span>
            </div>
          </div>
          <button
            type="button"
            className="history-btn"
            onClick={() => onNavigate && onNavigate(item.phaseId)}
          >
            {item.btnText} →
          </button>
        </div>
      ))}
    </div>
  )
}
