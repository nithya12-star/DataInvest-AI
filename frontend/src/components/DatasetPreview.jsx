// components/DatasetPreview.jsx — first-N rows table
import { useState } from 'react'
import './DatasetPreview.css'

const ROWS_OPTIONS = [5, 10, 20, 50]

export default function DatasetPreview({ profile }) {
  const [nRows, setNRows] = useState(10)
  const { overview } = profile

  // Reconstruct preview from profile data (we receive the raw preview rows separately)
  const rows = profile.preview_rows || []
  const cols = overview.column_names

  const visibleRows = rows.slice(0, nRows)

  return (
    <section className="section-block fade-in">
      <div className="section-header-row">
        <div className="section-title">
          <span>Dataset Preview</span>
        </div>
        <select
          className="rows-select"
          value={nRows}
          onChange={e => setNRows(Number(e.target.value))}
        >
          {ROWS_OPTIONS.map(n => (
            <option key={n} value={n}>Show {n} rows</option>
          ))}
        </select>
      </div>

      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th className="row-num-col">#</th>
              {cols.map(col => (
                <th key={col} title={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.length === 0 ? (
              <tr><td colSpan={cols.length + 1} style={{textAlign:'center',color:'var(--text-muted)',padding:'20px'}}>No preview data available.</td></tr>
            ) : (
              visibleRows.map((row, i) => (
                <tr key={i}>
                  <td className="row-num-col text-sm">{i + 1}</td>
                  {cols.map(col => (
                    <td key={col} title={String(row[col] ?? '')}>
                      {row[col] === null || row[col] === undefined
                        ? <span className="null-cell">—</span>
                        : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      <div className="preview-footer">
        Showing {Math.min(nRows, rows.length)} of {overview.n_rows.toLocaleString()} rows
        &nbsp;·&nbsp; {overview.n_cols} columns
      </div>
    </section>
  )
}
