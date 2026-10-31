// frontend/src/components/VegaChart.jsx — Vega-Lite Visual Evidence Renderer
import { useEffect, useRef, useState } from 'react'
import vegaEmbed from 'vega-embed'
import './VegaChart.css'

export default function VegaChart({ visualEvidence, subtitle = null }) {
  const containerRef = useRef(null)
  const viewRef = useRef(null)
  const [embedError, setEmbedError] = useState(null)

  const figureSpec = visualEvidence?.figure_spec
  const diagnostics = visualEvidence?.visual_diagnostics
  const figureType = visualEvidence?.figure_type || 'chart'
  const title = visualEvidence?.title || 'Visual Evidence'

  // Safely parse/sanitize spec at render-time without cascading setState
  let sanitizedSpec = null
  let specParseError = null
  if (figureSpec) {
    try {
      if (typeof figureSpec === 'string') {
        sanitizedSpec = JSON.parse(figureSpec)
      } else {
        sanitizedSpec = JSON.parse(JSON.stringify(figureSpec))
      }
      sanitizedSpec.background = sanitizedSpec.background || 'transparent'
      if (!sanitizedSpec.width) {
        sanitizedSpec.width = 360
      }
      if (!sanitizedSpec.height) {
        sanitizedSpec.height = 220
      }
      sanitizedSpec.autosize = sanitizedSpec.autosize || { type: 'fit', contains: 'padding' }
    } catch (err) {
      specParseError = err.message
    }
  }

  useEffect(() => {
    let isCancelled = false

    if (!containerRef.current || !sanitizedSpec || specParseError) {
      return
    }

    // Clean up previous view instance
    if (viewRef.current) {
      try {
        viewRef.current.finalize()
      } catch {
        // Safe disposal
      }
      viewRef.current = null
    }

    const embedOpts = {
      actions: false,
      mode: 'vega-lite',
      renderer: 'svg',
      theme: 'dark',
      defaultStyle: true,
    }

    vegaEmbed(containerRef.current, sanitizedSpec, embedOpts)
      .then((res) => {
        if (isCancelled) {
          res.view.finalize()
          return
        }
        viewRef.current = res.view
      })
      .catch((err) => {
        if (!isCancelled) {
          setEmbedError(err.message || 'Failed to render Vega specification.')
        }
      })

    return () => {
      isCancelled = true
      if (viewRef.current) {
        try {
          viewRef.current.finalize()
        } catch {
          // Safe disposal
        }
        viewRef.current = null
      }
    }
  }, [sanitizedSpec, specParseError])

  if (!visualEvidence) {
    return null
  }

  const activeError = specParseError
    ? `Malformed chart specification: ${specParseError}`
    : embedError

  return (
    <div className="vega-evidence-container">
      {/* Header */}
      <div className="vega-evidence-header">
        <div className="vega-evidence-title-group">
          <span className="vega-evidence-title">📊 {title}</span>
          <span className="vega-evidence-type-badge">{figureType}</span>
        </div>
        {subtitle && <span className="vega-evidence-subtitle">{subtitle}</span>}
      </div>

      {/* Main Chart Canvas or Fallback */}
      <div className="vega-embed-wrapper">
        {activeError ? (
          <div className="vega-chart-fallback">
            ⚠️ <strong>Visual rendering unavailable:</strong> {activeError}
            <div style={{ marginTop: '4px', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              Raw diagnostic summary preserved below.
            </div>
          </div>
        ) : !figureSpec ? (
          <div className="vega-chart-fallback">
            ℹ No chart specification available for this task.
          </div>
        ) : (
          <div ref={containerRef} className="vega-embed-canvas" />
        )}
      </div>

      {/* Programmatic Diagnostics */}
      {diagnostics && (
        <div className="vega-diagnostics-bar">
          {diagnostics.outlier_points_count !== undefined && diagnostics.outlier_points_count !== null && (
            <span className="vega-diag-chip">
              <span>Points / Outliers:</span>
              <strong>{diagnostics.outlier_points_count}</strong>
            </span>
          )}
          {diagnostics.linearity_indicator && (
            <span className="vega-diag-chip accent">
              <span>Linearity:</span>
              <strong>{diagnostics.linearity_indicator}</strong>
            </span>
          )}
          {diagnostics.distribution_shape && (
            <span className="vega-diag-chip">
              <span>Shape:</span>
              <strong>{diagnostics.distribution_shape}</strong>
            </span>
          )}
        </div>
      )}

      {/* Programmatic visual notes */}
      {diagnostics?.visual_notes && diagnostics.visual_notes.length > 0 && (
        <ul className="vega-notes-list">
          {diagnostics.visual_notes.map((note, idx) => (
            <li key={idx}>{note}</li>
          ))}
        </ul>
      )}

      {/* Mandatory Limitation Notice */}
      <div className="vega-limitation-notice">
        <span className="vega-limitation-icon">ℹ</span>
        <span>
          <strong>Limitation Notice:</strong> Programmatic visual diagnostics used.
          No vision-capable model inspected or interpreted this graph.
        </span>
      </div>
    </div>
  )
}
