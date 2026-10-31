// components/Hero.jsx
import './Hero.css'

export default function Hero() {
  return (
    <section className="hero">
      <div className="hero-eyebrow">
        <span>◆</span> AI Dataset Intelligence
      </div>
      <h1 className="hero-title">
        Investigate Your <span>Data</span>
      </h1>
      <p className="hero-subtitle">
        Upload a CSV and automatically understand its structure, quality, and statistics.
      </p>
    </section>
  )
}
