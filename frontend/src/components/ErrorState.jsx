// components/ErrorState.jsx
import './ErrorState.css'

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="error-state fade-in">
      <div className="error-card card card-no-hover">
        <div className="error-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
        </div>
        <div className="error-title">Upload Failed</div>
        <div className="error-message">{message}</div>
        <button className="btn btn-outline" onClick={onRetry}>Try Again</button>
      </div>
    </div>
  )
}
