/**
 * ImageBrowser Component
 * Browse and select images from Kaggle dataset (test/val splits only)
 * Prevents data leakage by excluding training images
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ImageBrowser.css';

export default function ImageBrowser({ onImageSelected, onClose }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [split, setSplit] = useState('test');
  const [selectedFile, setSelectedFile] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [filterGrade, setFilterGrade] = useState(null);
  const [thumbnails, setThumbnails] = useState({});

  // Fetch available images on component mount or when split changes
  useEffect(() => {
    fetchImages();
    fetchStats();
    setThumbnails({});
  }, [split]);

  // Lazily load thumbnails whenever the image list changes
  useEffect(() => {
    if (!images.length) return;
    const cancelRef = { cancelled: false };
    const loadThumbs = async () => {
      for (const img of images) {
        if (cancelRef.cancelled) break;
        if (thumbnails[img.kaggle_path]) continue;
        try {
          const r = await axios.get('/api/image-thumbnail', {
            params: { kaggle_path: img.kaggle_path }
          });
          if (!cancelRef.cancelled) {
            setThumbnails(prev => ({ ...prev, [img.kaggle_path]: r.data.image_base64 }));
          }
        } catch {
          // ignore thumbnail failures
        }
      }
    };
    loadThumbs();
    return () => { cancelRef.cancelled = true; };
  }, [images]);

  const fetchImages = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/available-images', {
        params: { split, limit: 100 }
      });
      setImages(response.data.images || []);
    } catch (err) {
      setError(`Failed to load images: ${err.message}`);
      console.error('Error fetching images:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/dataset-stats', {
        params: { split }
      });
      setStats(response.data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const filteredImages = filterGrade !== null 
    ? images.filter(img => img.kl_grade === filterGrade)
    : images;

  const handleImageSelect = async (image) => {
    setSelectedFile(image);
    setPredicting(true);
    setError(null);

    try {
      // Call the predict-from-dataset endpoint
      const response = await axios.post('/api/predict-from-dataset', null, {
        params: {
          filename: image.filename,
          split: split
        }
      });

      setResult(response.data);
      if (onImageSelected) {
        onImageSelected(response.data);
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message;
      setError(`Prediction failed: ${errorMsg}`);
      console.error('Prediction error:', err);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="image-browser-overlay">
      <div className="image-browser-modal">
        {/* Header */}
        <div className="browser-header">
          <h2>📁 Browse Kaggle Dataset</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        {/* Dataset Source Info */}
        <div className="browser-source-banner">
          <div className="source-icon">&#x1F4CA;</div>
          <div className="source-text">
            <strong>Kaggle &mdash; Knee Osteoarthritis Dataset with Severity</strong>
            <span>
              Published by <em>shashwatwork</em> &middot;
              <a
                href="https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity"
                target="_blank"
                rel="noreferrer"
              >kaggle.com/datasets/shashwatwork/&hellip;</a>
              &middot; Anteroposterior (AP) plain-film X-ray images labelled KL-0 through KL-4
              &middot; <span className="source-info-safe">&#x1F512; Training split is blocked to prevent data leakage &mdash; only test&nbsp;/&nbsp;val images are shown</span>
            </span>
          </div>
        </div>

        {/* Controls */}
        <div className="browser-controls">
          <div className="control-group">
            <label>Data Split:</label>
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="test">Test Set</option>
              <option value="val">Validation Set</option>
            </select>
            <span className="safety-badge">🔒 No training data</span>
          </div>

          <div className="control-group">
            <label>Filter by KL Grade:</label>
            <select 
              value={filterGrade !== null ? filterGrade : 'all'}
              onChange={(e) => setFilterGrade(e.target.value === 'all' ? null : parseInt(e.target.value))}
            >
              <option value="all">All Grades</option>
              <option value="0">KL-0 (Normal)</option>
              <option value="1">KL-1 (Doubtful)</option>
              <option value="2">KL-2 (Mild)</option>
              <option value="3">KL-3 (Moderate)</option>
              <option value="4">KL-4 (Severe)</option>
            </select>
          </div>

          {stats && (
            <div className="stats-display">
              <span>Total: {stats.total_images} images</span>
              <span>Size: {(stats.total_size_kb / 1024).toFixed(1)} MB</span>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading images...</p>
          </div>
        )}

        {/* Image Grid */}
        {!loading && filteredImages.length > 0 && (
          <div className="image-grid">
            {filteredImages.map((image, idx) => (
              <div
                key={idx}
                className={`image-card ${selectedFile?.filename === image.filename ? 'selected' : ''}`}
                onClick={() => handleImageSelect(image)}
                style={{
                  cursor: predicting && selectedFile?.filename === image.filename ? 'wait' : 'pointer',
                  opacity: predicting && selectedFile?.filename === image.filename ? 0.6 : 1
                }}
              >
                <div className="image-placeholder">
                  <div className={`grade-badge grade-${image.kl_grade}`}>KL-{image.kl_grade}</div>
                  {thumbnails[image.kaggle_path]
                    ? <img
                        src={`data:image/png;base64,${thumbnails[image.kaggle_path]}`}
                        alt={image.filename}
                        style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '4px' }}
                      />
                    : <div className="thumb-loading">⏳</div>
                  }
                </div>
                <div className="image-info">
                  <p className="filename">{image.filename.substring(0, 20)}...</p>
                  <p className="size">{(image.file_size / 1024).toFixed(1)} KB</p>
                </div>
                {predicting && selectedFile?.filename === image.filename && (
                  <div className="predicting-overlay">
                    <div className="mini-spinner"></div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {!loading && filteredImages.length === 0 && (
          <div className="empty-state">
            <p>No images found for the selected filter</p>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className="result-container">
            <div className="result-header">✓ Prediction Complete</div>
            <div className="result-body">
              <div className="result-row">
                <span>File:</span>
                <strong>{result.filename}</strong>
              </div>
              <div className="result-row">
                <span>Predicted:</span>
                <strong>{result.kl_label}</strong>
              </div>
              <div className="result-row">
                <span>Confidence:</span>
                <strong>{(result.confidence * 100).toFixed(1)}%</strong>
              </div>
              <div className="result-row">
                <span>Risk Level:</span>
                <span className={`risk-badge ${result.traffic_light}`}>
                  {result.risk_level}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
