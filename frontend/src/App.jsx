import './App.css'
import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import ResultsDisplay from './components/ResultsDisplay'
import ImageBrowser from './components/ImageBrowser'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('upload') // 'upload' | 'dataset'
  const [showBrowser, setShowBrowser] = useState(false)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch('/api/predict', { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`Server error (${response.status})`)
      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleImageFromDataset = (result) => {
    setShowBrowser(false)
    setPrediction(result)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleReset = () => {
    setPrediction(null)
    setError(null)
    setActiveTab('upload')
  }

  const handleTabChange = (tab) => {
    setActiveTab(tab)
    if (tab === 'dataset') setShowBrowser(true)
    else setShowBrowser(false)
  }

  return (
    <div className="app-shell">
      {/* ── Top Navigation ── */}
      <nav className="app-navbar">
        <div className="navbar-brand">
          <div className="navbar-icon">🦴</div>
          <div className="navbar-title">
            <strong>KOA Clinical Decision Support</strong>
            <span>Knee Osteoarthritis AI · Cascade Pipeline v1.0</span>
          </div>
        </div>
        <div className="navbar-actions">
          <div className="nav-status">System Online</div>
        </div>
      </nav>

      {/* ── Mode Tabs (only when no result) ── */}
      {!prediction && (
        <div className="app-tabs">
          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => handleTabChange('upload')}
          >
            <span className="tab-icon">📤</span>
            Upload X-ray
          </button>
          <button
            className={`tab-btn ${activeTab === 'dataset' ? 'active' : ''}`}
            onClick={() => handleTabChange('dataset')}
          >
            <span className="tab-icon">🗂</span>
            Browse Kaggle Dataset
          </button>
        </div>
      )}

      {/* ── Page Content ── */}
      <div className="app-content">
        {prediction ? (
          <ResultsDisplay prediction={prediction} onReset={handleReset} />
        ) : activeTab === 'upload' ? (
          <ImageUploader onUpload={handleUpload} loading={loading} error={error} />
        ) : (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--slate-400)' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🗂</div>
            <p style={{ fontSize: '1rem', marginBottom: '8px', color: 'var(--slate-600)' }}>
              Kaggle Dataset Browser is opening…
            </p>
            <p style={{ fontSize: '0.85rem' }}>Select an image from the modal to run prediction</p>
          </div>
        )}
      </div>

      {/* ── Dataset Browser Modal ── */}
      {showBrowser && (
        <ImageBrowser
          onImageSelected={handleImageFromDataset}
          onClose={() => { setShowBrowser(false); setActiveTab('upload') }}
        />
      )}

      {/* ── Footer ── */}
      <footer className="app-footer">
        <div className="footer-disclaimer">
          <span className="footer-disclaimer-icon">⚕️</span>
          <p>
            <strong>Clinical Disclaimer:</strong> This system is intended for screening assistance only
            and does not replace clinical judgment or professional medical advice.
            Always consult a qualified healthcare provider for diagnosis and treatment.
          </p>
        </div>
        <div className="footer-meta">
          <span>&#x25B8; ConvNeXt-L · 87.8% Accuracy</span>
          <span>&#x25B8; Cascade Pipeline v1.0</span>
          <span>&#x25B8; Kaggle Dataset: KL Grade 0–4</span>
        </div>
      </footer>
    </div>
  )
}

export default App
