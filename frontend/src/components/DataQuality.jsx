// components/DataQuality.jsx — data quality summary section
import './DataQuality.css'

function QualityRow({ icon, label, value, status, note }) {
  return (
    <div className={`quality-row quality-row--${status}`}>
      <div className="quality-icon">{icon}</div>
      <div className="quality-body">
        <div className="quality-label">{label}</div>
        {note && <div className="quality-note">{note}</div>}
      </div>
      <div className="quality-value">{value}</div>
    </div>
  )
}

export default function DataQuality({ profile }) {
  const { missing, duplicates, cardinality, outliers } = profile

  const totalOutliers = Object.values(outliers || {}).reduce(
    (sum, o) => sum + (o.n_potential_outliers || 0), 0
  )
  const missingPct = missing.total_missing_pct
  const dupPct = duplicates.duplicate_pct
  const constCols = cardinality.constant_columns || []
  const highCard = cardinality.high_cardinality_columns || []

  const missingStatus = missingPct === 0 ? 'success' : missingPct < 5 ? 'warning' : 'error'
  const dupStatus     = duplicates.n_duplicates === 0 ? 'success' : 'warning'
  const constStatus   = constCols.length === 0 ? 'success' : 'warning'
  const highCardStatus = highCard.length === 0 ? 'success' : 'warning'
  const outlierStatus  = totalOutliers === 0 ? 'success' : 'warning'

  return (
    <section className="section-block fade-in">
      <div className="section-title">Data Quality</div>
      <div className="quality-list">
        <QualityRow
          icon={missingStatus === 'success' ? '✓' : '⚠'}
          label="Missing Values"
          value={missing.total_missing === 0
            ? 'None'
            : `${missing.total_missing.toLocaleString()} (${missingPct}%)`}
          status={missingStatus}
          note={missing.total_missing > 0 ? `Across ${Object.values(missing.per_column).filter(m=>m.missing_count>0).length} column(s)` : null}
        />
        <QualityRow
          icon={dupStatus === 'success' ? '✓' : '⚠'}
          label="Duplicate Rows"
          value={duplicates.n_duplicates === 0 ? 'None' : `${duplicates.n_duplicates.toLocaleString()} (${dupPct}%)`}
          status={dupStatus}
        />
        <QualityRow
          icon={constStatus === 'success' ? '✓' : '⚠'}
          label="Constant Columns"
          value={constCols.length === 0 ? 'None' : constCols.length}
          status={constStatus}
          note={constCols.length > 0 ? constCols.join(', ') : null}
        />
        <QualityRow
          icon={highCardStatus === 'success' ? '✓' : '⚠'}
          label="High-Cardinality Columns"
          value={highCard.length === 0 ? 'None' : highCard.length}
          status={highCardStatus}
          note={highCard.length > 0 ? highCard.slice(0,4).join(', ') + (highCard.length > 4 ? '…' : '') : null}
        />
        <QualityRow
          icon={outlierStatus === 'success' ? '✓' : '⚠'}
          label="Potential Outliers (IQR)"
          value={totalOutliers === 0 ? 'None detected' : `${totalOutliers.toLocaleString()} values`}
          status={outlierStatus}
          note={totalOutliers > 0 ? 'Values outside Q1−1.5·IQR or Q3+1.5·IQR — review required' : null}
        />
      </div>
    </section>
  )
}
