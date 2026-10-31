// components/LoadingState.jsx
import './LoadingState.css'

export default function LoadingState({ filename }) {
  return (
    <div className="loading-state fade-in">
      <div className="loading-card card card-no-hover">
        <div className="spinner"></div>
        <div className="loading-title">Profiling Dataset</div>
        <div className="loading-filename">{filename}</div>
        <div className="loading-sub">Running statistical analysis…</div>
      </div>
    </div>
  )
}
