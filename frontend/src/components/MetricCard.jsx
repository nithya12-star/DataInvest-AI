// components/MetricCard.jsx — compact stat card with icon
import './MetricCard.css'

export default function MetricCard({ label, value, sub, icon, variant }) {
  const variantClass = variant ? `metric-card--${variant}` : ''
  return (
    <div className={`metric-card card card-no-hover ${variantClass}`}>
      {icon && <div className="metric-icon">{icon}</div>}
      <div className="metric-body">
        <div className="metric-value">{value}</div>
        <div className="metric-label">{label}</div>
        {sub && <div className="metric-sub">{sub}</div>}
      </div>
    </div>
  )
}
