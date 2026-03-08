import './ResultsDisplay.css'
import Visualizer from './Visualizer'
import AgentReport from './AgentReport'

const TRAFFIC_ICONS = { green: '🟢', yellow: '🟡', red: '🔴' }
const RISK_LABELS   = { green: 'Low Risk — No Definite OA', yellow: 'Moderate OA Detected', red: 'Severe OA Detected' }

export default function ResultsDisplay({ prediction, onReset }) {
  const { kl_grade, kl_label, confidence, all_probabilities,
          traffic_light, risk_level, recommendation,
          grad_cam, clinical_summary, filename, split } = prediction

  const color   = traffic_light || 'green'
  const confPct = (confidence * 100).toFixed(1)

  const downloadReport = async () => {
    try {
      const res = await fetch('/api/generate-report', {
        method: 'POST',
        body: JSON.stringify({
          patient_id: 'UNKNOWN',
          include_grad_cam: true,
          kl_grade,
          kl_label,
          confidence,
          all_probabilities,
          risk_level,
          recommendation,
          traffic_light: color,
        }),
        headers: { 'Content-Type': 'application/json' },
      })
      if (!res.ok) {
        let detail = 'Report generation failed'
        try {
          const err = await res.json()
          const raw = err?.detail
          if (typeof raw === 'string') detail = raw
          else if (Array.isArray(raw)) detail = raw.map(e => e.msg || JSON.stringify(e)).join('; ')
          else if (raw) detail = JSON.stringify(raw)
        } catch { /* keep default */ }
        throw new Error(detail)
      }
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href = url; a.download = 'KOA_Report.pdf'
      document.body.appendChild(a); a.click()
      URL.revokeObjectURL(url); document.body.removeChild(a)
    } catch (e) { alert('Error downloading report: ' + e.message) }
  }

  // Find the top probability grade
  const topGrade = kl_label?.split(':')[0] ?? ''

  return (
    <div className="results-layout">
      {/* ── Status Banner ── */}
      <div className={`results-banner ${color}`}>
        <div className="banner-left">
          <div className={`traffic-dot ${color}`}>{TRAFFIC_ICONS[color]}</div>
          <div className="banner-info">
            <h2>{kl_label}</h2>
            <p>{RISK_LABELS[color]}{filename ? ` · ${filename}${split ? ` (${split})` : ''}` : ''}</p>
          </div>
        </div>
        <div className={`confidence-chip ${color}`}>
          <span className="conf-label">Confidence</span>
          <span className="conf-value">{confPct}%</span>
        </div>
      </div>

      {/* ── Detail Grid ── */}
      <div className="results-grid">
        {/* Confidence gauge */}
        <div className="section-card">
          <div className="section-head">
            <h3>Prediction Confidence</h3>
            <span className={`badge badge-${color === 'green' ? 'green' : color === 'yellow' ? 'amber' : 'red'}`}>
              {risk_level}
            </span>
          </div>
          <div className="section-body">
            <div className="conf-gauge-wrap">
              <div className="conf-track">
                <div className={`conf-fill ${color}`} style={{ width: `${confPct}%` }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--slate-400)' }}>0%</span>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: `var(--${color === 'green' ? 'green' : color === 'yellow' ? 'amber' : 'red'}-600)` }}>{confPct}%</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--slate-400)' }}>100%</span>
              </div>
            </div>

            {recommendation && (
              <div className="recommendation-card" style={{ marginTop: 'var(--spacing-lg)' }}>
                <span className="rec-icon">📋</span>
                <div className="rec-text">
                  <strong>Clinical Recommendation</strong>
                  {recommendation}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* All class probabilities */}
        <div className="section-card">
          <div className="section-head">
            <h3>KL Grade Probabilities</h3>
          </div>
          <div className="section-body">
            <div className="prob-list">
              {all_probabilities && Object.entries(all_probabilities).map(([kl, prob]) => {
                const pct = (prob * 100).toFixed(1)
                const isTop = kl === topGrade
                return (
                  <div className="prob-row" key={kl}>
                    <span className="kl-label">{kl}</span>
                    <div className="prob-track">
                      <div
                        className={`prob-fill ${isTop ? 'top' : 'normal'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className={`pct ${isTop ? 'top' : ''}`}>{pct}%</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Grad-CAM */}
        {grad_cam?.heatmap_base64 && (
          <div className="section-card results-full">
            <div className="section-head"><h3>Grad-CAM — Model Attention</h3></div>
            <div className="section-body">
              <Visualizer heatmap={grad_cam.heatmap_base64} regions={grad_cam.attention_regions} />
            </div>
          </div>
        )}

        {/* Clinical summary */}
        {clinical_summary && (
          <div className="section-card results-full">
            <div className="section-head"><h3>AI Clinical Summary</h3></div>
            <div className="section-body">
              <AgentReport summary={clinical_summary} />
            </div>
          </div>
        )}
      </div>

      {/* ── Actions ── */}
      <div className="results-actions">
        <button className="btn btn-primary" onClick={downloadReport}>
          ↓ &nbsp;Download PDF Report
        </button>
        <button className="btn btn-secondary" onClick={onReset}>
          ↺ &nbsp;Analyse Another Image
        </button>
      </div>
    </div>
  )
}
