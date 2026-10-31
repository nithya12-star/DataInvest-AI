// components/NumericalProfile.jsx — stats for numerical columns
import { useState } from 'react'
import './NumericalProfile.css'

function fmt(v, digits = 4) {
  if (v === null || v === undefined) return <span className="no-val">—</span>
  const n = Number(v)
  if (isNaN(n)) return String(v)
  return Math.abs(n) >= 1000
    ? n.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : n.toFixed(digits)
}

const STAT_COLS = [
  { key: 'n_valid',   label: 'Valid n',  className: '' },
  { key: 'mean',      label: 'Mean',     className: 'highlight' },
  { key: 'median',    label: 'Median',   className: 'highlight' },
  { key: 'std',       label: 'Std Dev',  className: '' },
  { key: 'min',       label: 'Min',      className: '' },
  { key: 'q1',        label: 'Q1',       className: '' },
  { key: 'q3',        label: 'Q3',       className: '' },
  { key: 'max',       label: 'Max',      className: '' },
  { key: 'iqr',       label: 'IQR',      className: 'highlight' },
  { key: 'skewness',  label: 'Skewness', className: '' },
]

export default function NumericalProfile({ profile }) {
  const { numerical_stats, outliers } = profile
  const cols = Object.keys(numerical_stats)
  const [selected, setSelected] = useState(cols[0] || '')

  if (cols.length === 0) {
    return (
      <section className="section-block fade-in">
        <div className="section-title">Numerical Statistics</div>
        <div className="empty-state">No numerical columns detected.</div>
      </section>
    )
  }

  const activeCol = cols.includes(selected) ? selected : cols[0]
  const stats = numerical_stats[activeCol] || {}
  const outInfo = outliers?.[activeCol] || {}


  return (
    <section className="section-block fade-in">
      <div className="section-header-row">
        <div className="section-title">Numerical Statistics</div>
        <select
          className="rows-select"
          value={selected}
          onChange={e => setSelected(e.target.value)}
        >
          {cols.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {/* Stat grid */}
      <div className="stat-grid">
        {STAT_COLS.map(({ key, label, className }) => (
          <div key={key} className={`stat-cell ${className}`}>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{fmt(stats[key])}</div>
          </div>
        ))}
      </div>

      {/* Outlier summary */}
      {outInfo.n_potential_outliers !== null && outInfo.n_potential_outliers !== undefined && (
        <div className={`outlier-summary ${outInfo.n_potential_outliers > 0 ? 'has-outliers' : 'no-outliers'}`}>
          <span className="outlier-icon">{outInfo.n_potential_outliers > 0 ? '⚠' : '✓'}</span>
          <span>
            {outInfo.n_potential_outliers > 0
              ? <><strong>{outInfo.n_potential_outliers}</strong> potential outlier{outInfo.n_potential_outliers !== 1 ? 's' : ''} ({outInfo.outlier_pct}%) · IQR fences: [{outInfo.lower_fence}, {outInfo.upper_fence}]</>
              : 'No potential outliers detected (IQR 1.5× rule)'}
          </span>
        </div>
      )}

      {/* All columns summary table */}
      <details className="all-cols-details">
        <summary>All numerical columns</summary>
        <div className="data-table-wrap" style={{marginTop:'12px'}}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Column</th>
                {STAT_COLS.slice(1).map(s => <th key={s.key}>{s.label}</th>)}
                <th>Outliers</th>
              </tr>
            </thead>
            <tbody>
              {cols.map(col => {
                const s = numerical_stats[col] || {}
                const o = outliers?.[col] || {}
                return (
                  <tr key={col} className={col === selected ? 'active-row' : ''}>
                    <td className="col-name-cell">{col}</td>
                    {STAT_COLS.slice(1).map(({ key, className }) => (
                      <td key={key} className={className}>{fmt(s[key])}</td>
                    ))}
                    <td className={o.n_potential_outliers > 0 ? 'cell-warning' : ''}>
                      {o.n_potential_outliers ?? '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  )
}
