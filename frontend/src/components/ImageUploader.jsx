import { useState } from 'react'
import './ImageUploader.css'

export default function ImageUploader({ onUpload, loading, error }) {
  const [dragOver, setDragOver] = useState(false)
  const [fileName, setFileName] = useState('')

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file?.type.startsWith('image/')) { setFileName(file.name); onUpload(file) }
  }

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    if (file) { setFileName(file.name); onUpload(file) }
  }

  return (
    <div className="uploader-layout">
      {/* ── Drop Zone ── */}
      <div className="drop-zone-card">
        <div className="drop-zone-header">
          <h2>Upload X-ray Image</h2>
          <span className="badge badge-blue">Stage 1 + 2 Analysis</span>
        </div>
        <div className="drop-zone-body">
          <div
            className={`drop-target${dragOver ? ' drag-over' : ''}`}
            onDragEnter={handleDrag} onDragLeave={handleDrag}
            onDragOver={handleDrag} onDrop={handleDrop}
          >
            {loading ? (
              <div className="loading-state">
                <div className="spinner" />
                <p>Analysing image — please wait…</p>
              </div>
            ) : (
              <>
                <div className="drop-icon">🦴</div>
                <h3>Drop your knee X-ray here</h3>
                <p>or click to browse from your device</p>
                <input type="file" id="file-input" onChange={handleChange} accept="image/*" disabled={loading} />
                <label htmlFor="file-input" className="browse-btn">
                  Browse Files
                </label>
                <p className="formats-hint">Supported formats: JPG · PNG · DICOM · WebP</p>
              </>
            )}
          </div>

          {fileName && !loading && (
            <div className="file-selected">
              ✓ &nbsp;<strong>{fileName}</strong>
            </div>
          )}

          {error && (
            <div className="error-banner">
              ⚠ &nbsp;<span><strong>Error:</strong> {error}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Info Panel ── */}
      <div className="info-panel">
        {/* Pipeline steps */}
        <div className="info-card">
          <div className="info-card-title">How it works</div>
          <div className="pipeline-steps">
            {[
              ['Stage 1 — Screening', 'KL ≥ 1 detection. Separates Healthy / Subtle from Definite OA.'],
              ['Stage 2 — Severity', 'KL ≥ 3 classification. Distinguishes Moderate from Severe OA.'],
              ['Grad-CAM Heatmap', 'Visualises which joint regions drove the prediction.'],
              ['Clinical Summary', 'AI agent generates evidence-based recommendation.'],
            ].map(([title, desc], i) => (
              <div className="pipeline-step" key={i}>
                <div className="step-num">{i + 1}</div>
                <div className="step-text"><strong>{title}</strong>{desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Accuracy metrics */}
        <div className="info-card">
          <div className="info-card-title">Model Performance</div>
          {[
            ['Architecture', 'ConvNeXt-L', false],
            ['Stage 1 Accuracy', '87.8 %', true],
            ['Stage 1 AUC', '0.952', false],
            ['Stage 2 Accuracy', '96.1 %', true],
            ['Stage 2 AUC', '0.990', false],
            ['vs. 5-class baseline', '+23.2 %', true],
          ].map(([label, value, highlight]) => (
            <div className="accuracy-row" key={label}>
              <span className="label">{label}</span>
              <span className={`value${highlight ? ' highlight' : ''}`}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
