// components/ExperimentModeToggle.jsx — 3-way toggle between Compare Both, Baseline, and Proposed modes
import './ExperimentModeToggle.css'

export default function ExperimentModeToggle({
  activeMode = 'compare',
  onChangeMode,
  baselineScore = null,
  proposedScore = null,
  overallDelta = null,
  baselineGrade = null,
  proposedGrade = null,
}) {
  return (
    <div className="experiment-mode-bar fade-in">
      <div className="mode-bar-header">
        <div className="mode-bar-title-wrap">
          <span className="mode-bar-pill">Benchmark Mode Switcher</span>
          <span className="mode-bar-hint">Switch views without re-running computations</span>
        </div>
        {overallDelta !== null && overallDelta !== undefined && (
          <div className={`mode-delta-badge ${overallDelta >= 0 ? 'delta-positive' : 'delta-negative'}`}>
            <span>Empirical Delta:</span>
            <strong>{overallDelta >= 0 ? `+${overallDelta.toFixed(1)}` : overallDelta.toFixed(1)} pts</strong>
          </div>
        )}
      </div>

      <div className="mode-toggle-group" role="tablist" aria-label="Experiment Views">
        <button
          type="button"
          role="tab"
          aria-selected={activeMode === 'compare'}
          className={`mode-toggle-btn ${activeMode === 'compare' ? 'active compare-active' : ''}`}
          onClick={() => onChangeMode('compare')}
        >
          <span className="mode-icon">⚖</span>
          <span className="mode-label">Compare Both (Benchmark)</span>
          <span className="mode-tag-pill">Side-by-Side</span>
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeMode === 'baseline'}
          className={`mode-toggle-btn ${activeMode === 'baseline' ? 'active baseline-active' : ''}`}
          onClick={() => onChangeMode('baseline')}
        >
          <span className="mode-icon">🔬</span>
          <span className="mode-label">Baseline Architecture</span>
          {baselineScore !== null && baselineScore !== undefined && (
            <span className="mode-score-badge">
              {baselineGrade ? `${baselineGrade} · ` : ''}{typeof baselineScore === 'number' ? baselineScore.toFixed(1) : baselineScore} pts
            </span>
          )}
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeMode === 'proposed'}
          className={`mode-toggle-btn ${activeMode === 'proposed' ? 'active proposed-active' : ''}`}
          onClick={() => onChangeMode('proposed')}
        >
          <span className="mode-icon">🛡</span>
          <span className="mode-label">Proposed Architecture</span>
          {proposedScore !== null && proposedScore !== undefined && (
            <span className="mode-score-badge proposed-score">
              {proposedGrade ? `${proposedGrade} · ` : ''}{typeof proposedScore === 'number' ? proposedScore.toFixed(1) : proposedScore} pts
            </span>
          )}
        </button>
      </div>
    </div>
  )
}
