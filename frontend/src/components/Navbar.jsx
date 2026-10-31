// components/Navbar.jsx — Dynamic navigation bar with active phase badge
import { getPhaseById } from '../constants/phases'
import './Navbar.css'

export default function Navbar({
  onUploadClick,
  hasData,
  currentPhase = 1,
  runCount = 0,
  onOpenHistory,
  onOpenBenchmark,
}) {
  const activePhaseInfo = getPhaseById(currentPhase)

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <a href="/" className="navbar-brand">
          <div className="navbar-logo-icon">📊</div>
          <span className="navbar-brand-name">
            DataInvest <span>AI</span>
          </span>
        </a>
        <div className="navbar-right">
          {hasData && (
            <span className="navbar-phase" title={activePhaseInfo.description}>
              {activePhaseInfo.title}
            </span>
          )}

          <button
            type="button"
            className="btn btn-outline btn-history"
            onClick={onOpenHistory}
            title="View saved experiment runs and evaluation results"
          >
            <span>📁</span>
            <span>History</span>
            {runCount > 0 && <span className="history-count-badge">{runCount}</span>}
          </button>

          {!hasData && (
            <button className="btn btn-primary" onClick={onUploadClick}>
              Upload Dataset
            </button>
          )}
          {hasData && (
            <button className="btn btn-outline" onClick={onUploadClick}>
              ↑ New Dataset
            </button>
          )}
        </div>
      </div>
    </nav>
  )
}


