import './AgentReport.css'

export default function AgentReport({ summary }) {
  if (!summary) return null

  const lines = summary.split('\n').filter(l => l.trim())

  return (
    <div className="agent-summary">
      {lines.map((line, i) => {
        if (line.startsWith('##')) {
          return (
            <div className="summary-line heading" key={i}>
              {line.replace(/^#+\s*/, '')}
            </div>
          )
        }
        if (line.match(/^\*\*(.+):\*\*\s*(.+)/)) {
          const [, label, value] = line.match(/^\*\*(.+):\*\*\s*(.+)/)
          return (
            <div className="summary-line bold" key={i}>
              <span className="line-bullet">›</span>
              <span><strong>{label}:</strong>&nbsp;{value}</span>
            </div>
          )
        }
        if (line.startsWith('**') && line.endsWith('**')) {
          return (
            <div className="summary-line bold" key={i}>
              {line.replace(/\*\*/g, '')}
            </div>
          )
        }
        if (line.startsWith('-') || line.startsWith('*')) {
          return (
            <div className="summary-line list-item" key={i}>
              <span className="line-bullet">–</span>
              <span>{line.replace(/^[-*]\s*/, '')}</span>
            </div>
          )
        }
        return (
          <div className="summary-line" key={i}>{line}</div>
        )
      })}
      <div className="disclaimer-strip">
        <span>&#x26A0;</span>
        <span>AI-generated summary. For screening purposes only. Does not replace professional clinical evaluation.</span>
      </div>
    </div>
  )
}
