import './Visualizer.css'

export default function Visualizer({ heatmap, regions }) {
  if (!heatmap) return null

  return (
    <div className="viz-wrap">
      <div>
        <div className="viz-heatmap">
          {/* backend already returns a full data:image/png;base64,... URL */}
          <img
            src={heatmap}
            alt="Grad-CAM attention heatmap"
          />
        </div>
        <p className="viz-caption">
          Red = high model attention &middot; Blue = low attention
        </p>
      </div>

      <div className="viz-regions">
        {regions && regions.length > 0 ? regions.map((r, i) => (
          <div className="region-row" key={i}>
            <div className="region-meta">
              <span className="region-name">{r.region}</span>
              <span className="region-pct">{(r.intensity * 100).toFixed(0)}%</span>
            </div>
            <div className="region-track">
              <div className="region-fill" style={{ width: `${r.intensity * 100}%` }} />
            </div>
          </div>
        )) : (
          <p className="no-regions">No attention region data available.</p>
        )}
      </div>
    </div>
  )
}
