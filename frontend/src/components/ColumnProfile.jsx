// components/ColumnProfile.jsx — per-column type/missing/unique stats
import './ColumnProfile.css'

const TYPE_BADGE = {
  numerical:   { label: 'Numerical',   cls: 'badge-cyan' },
  categorical: { label: 'Categorical', cls: 'badge badge-type-cat' },
  datetime:    { label: 'Datetime',    cls: 'badge badge-type-dt' },
}

function fmt(n) {
  if (n === null || n === undefined) return '—'
  return typeof n === 'number' ? n.toLocaleString() : String(n)
}

export default function ColumnProfile({ profile }) {
  const { overview, column_types, missing, cardinality } = profile
  const cols = overview.column_names

  return (
    <section className="section-block fade-in">
      <div className="section-title">Column Profile</div>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Dtype</th>
              <th>Non-Null</th>
              <th>Missing</th>
              <th>Missing %</th>
              <th>Unique</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {cols.map(col => {
              const type = column_types[col] || 'categorical'
              const badge = TYPE_BADGE[type] || TYPE_BADGE.categorical
              const m = missing.per_column[col] || {}
              const c = cardinality.per_column[col] || {}
              const nonNull = overview.n_rows - (m.missing_count || 0)

              return (
                <tr key={col}>
                  <td className="col-name-cell" title={col}>{col}</td>
                  <td>
                    <span className={`badge ${badge.cls}`}>{badge.label}</span>
                  </td>
                  <td className="text-sm">{overview.dtypes[col] || '—'}</td>
                  <td className="highlight">{fmt(nonNull)}</td>
                  <td className={m.missing_count > 0 ? 'cell-warning' : ''}>{fmt(m.missing_count)}</td>
                  <td className={m.missing_pct > 0 ? 'cell-warning' : ''}>{m.missing_pct != null ? `${m.missing_pct}%` : '—'}</td>
                  <td className="highlight">{fmt(c.n_unique)}</td>
                  <td>
                    <div className="flags-cell">
                      {c.is_constant && <span className="badge badge-warning">Constant</span>}
                      {c.is_high_cardinality && <span className="badge badge-type-cat">High Card</span>}
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
