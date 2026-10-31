// components/InvestigationStepper.jsx — Investigation Journey Navigation Stepper
import { useState, useRef, useEffect, useCallback } from 'react'
import { PHASES } from '../constants/phases'
import './InvestigationStepper.css'

export default function InvestigationStepper({
  phases = PHASES,
  currentPhase = 1,
  maxUnlockedPhase = 1,
  completedPhases = [1],
  onSelectPhase,
}) {
  const navRef = useRef(null)
  const [canScrollRight, setCanScrollRight] = useState(false)

  const checkScroll = useCallback(() => {
    const el = navRef.current
    if (!el) return
    const hasMore = el.scrollWidth - el.clientWidth - el.scrollLeft > 12
    setCanScrollRight(hasMore)
  }, [])

  useEffect(() => {
    checkScroll()
    window.addEventListener('resize', checkScroll)
    return () => window.removeEventListener('resize', checkScroll)
  }, [checkScroll, phases])

  return (
    <div className="stepper-container" role="navigation" aria-label="Investigation Journey">
      <div className="stepper-scroll-wrapper">
        <div className="stepper-nav" ref={navRef} onScroll={checkScroll}>
        {phases.map((phase, index) => {
          const isActive = phase.id === currentPhase
          const isCompleted = completedPhases.includes(phase.id) && !isActive
          const isUnlocked = phase.id <= maxUnlockedPhase && !isActive && !isCompleted
          const isLocked = phase.id > maxUnlockedPhase

          let stateClass = 'state-locked'
          let statusText = 'Locked'
          let badgeContent = phase.stepNum

          if (isActive) {
            stateClass = 'state-active'
            statusText = 'Active'
            badgeContent = phase.stepNum
          } else if (isCompleted) {
            stateClass = 'state-completed'
            statusText = 'Completed'
            badgeContent = '✓'
          } else if (isUnlocked) {
            stateClass = 'state-unlocked'
            statusText = 'Ready'
            badgeContent = phase.stepNum
          } else if (isLocked) {
            stateClass = 'state-locked'
            statusText = 'Locked'
            badgeContent = '🔒'
          }

          return (
            <div key={phase.id} className={`stepper-item ${stateClass}`}>
              <button
                type="button"
                className="stepper-btn"
                onClick={() => {
                  if (!isLocked && onSelectPhase) {
                    onSelectPhase(phase.id)
                  }
                }}
                disabled={isLocked}
                aria-current={isActive ? 'step' : undefined}
                aria-label={`Step ${phase.id}: ${phase.label} (${statusText})`}
              >
                <div className="step-badge">{badgeContent}</div>
                <div className="step-content">
                  <span className="step-label">{phase.label}</span>
                  <span className="step-status-text">{statusText}</span>
                </div>
              </button>

              {index < phases.length - 1 && (
                <div className="stepper-divider" aria-hidden="true">
                  →
                </div>
              )}
            </div>
          )
        })}
        </div>
        {canScrollRight && (
          <div
            className="stepper-scroll-cue"
            aria-hidden="true"
            title="More phases available horizontally"
          >
            <span className="scroll-arrow">→</span>
          </div>
        )}
      </div>
    </div>
  )
}
