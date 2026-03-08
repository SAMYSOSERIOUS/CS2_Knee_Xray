import './TrafficLight.css'

export default function TrafficLight({ color, diagnosis }) {
  const lights = {
    green: { emoji: '🟢', label: 'LOW RISK', color: '#2ecc71' },
    yellow: { emoji: '🟡', label: 'MODERATE OA', color: '#f39c12' },
    red: { emoji: '🔴', label: 'SEVERE OA', color: '#e74c3c' },
  }

  const light = lights[color] || lights.green

  return (
    <div className={`traffic-light ${color}`}>
      <div className="light-display">
        <div className="light-emoji">{light.emoji}</div>
        <div className="light-label">{light.label}</div>
      </div>
      <div className="diagnosis-box">
        <p className="diagnosis-text">{diagnosis}</p>
      </div>
    </div>
  )
}
