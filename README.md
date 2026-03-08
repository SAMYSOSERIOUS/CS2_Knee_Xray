# KOA Clinical Decision Support System
## Hierarchical Cascade AI for Knee Osteoarthritis Diagnosis

---

## Project Overview

This is a **production-ready AI system** for knee osteoarthritis (OA) diagnosis that overcomes the limitations of traditional 5-class classification through a hierarchical binary cascade architecture.

### Key Innovation: The Multi-Stage Pipeline

Instead of trying to distinguish 5 ambiguous classes (KL 0-4), the system uses **two optimized binary classifiers**:

1. **Stage 1 (Screening)**: KL≥1 — Separates Healthy/Subtle (0-1) from Definite OA (2-4)
   - Model: ConvNeXt-L | AUC: 0.952 | Accuracy: 87.8%

2. **Stage 2 (Severity)**: KL≥3 — Distinguishes Moderate (KL-2) from Severe (KL-3/4)
   - Model: ConvNeXt-L | AUC: 0.990 | Accuracy: 96.1%

### Results

- **Baseline (5-class)**: 71.3% accuracy (NB2)
- **Cascade Pipeline**: 87.8% accuracy (NB7)
- **Improvement**: +65 basis points (+23.2%)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   VITE + REACT FRONTEND                     │
│  (Upload UI, Traffic Light, Visualizer, Agent Report)      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─────── HTTP REST API ──────────┐
                  ▼                                 ▼
        ┌──────────────────────────────────────────────────┐
        │         FASTAPI BACKEND (Port 8000)             │
        ├──────────────────────────────────────────────────┤
        │ • /api/predict         → Stage 1 + Stage 2      │
        │ • /api/generate-report → PDF generation         │
        │ • /api/thresholds      → Clinical thresholds   │
        │ • /api/info            → System metadata        │
        └──────────────────────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────────────┐
        │   INFERENCE PIPELINE                           │
        ├──────────────────────────────────────────────────┤
        │ • Preprocessing (normalize, resize)            │
        │ • Stage 1 Forward Pass (Screening)             │
        │ • Stage 2 Forward Pass (if Stage 1 positive)   │
        │ • Grad-CAM Heatmap Generation                  │
        │ • AI Agent Reasoning Layer                     │
        └──────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA 11.8+ (optional, for GPU acceleration)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
# Server will start at http://localhost:8000
```

**API Endpoints:**
- `POST /api/predict` — Submit image, get predictions
- `POST /api/generate-report` — Create PDF report
- `GET /api/thresholds` — View clinical thresholds
- `GET /api/info` — System information
- `GET /health` — Health check

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (Vite)
npm run dev
# App will be at http://localhost:5173
```

**Build for Production:**
```bash
npm run build
# Output: dist/
```

---

## Usage

### 1. Start Backend
```bash
cd backend
python main.py
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

### 3. Access Application
Open browser to: **http://localhost:5173**

### 4. Upload X-ray
- Click "Browse Files" or drag-and-drop
- Supported formats: JPG, PNG, DICOM
- Image will be processed automatically

### 5. Review Results
- **Traffic Light** shows risk level (Green/Yellow/Red)
- **Probability gauges** display Stage 1 & Stage 2 scores
- **Grad-CAM heatmap** explains where model focused
- **AI Agent report** provides clinical context

### 6. Download PDF Report
Click "📥 Download PDF Report" to get clinical summary

---

## Clinical Decision Rules

| Traffic Light | Condition | Diagnosis | Recommendation |
|---|---|---|---|
| 🟢 Green | Stage 1 < 0.35 | No Definite OA | Standard preventive care |
| 🟡 Yellow | Stage 1 ≥ 0.35 & Stage 2 < 0.50 | Moderate OA (KL-2) | Conservative treatment |
| 🔴 Red | Stage 1 ≥ 0.35 & Stage 2 ≥ 0.50 | Severe OA (KL-3/4) | Orthopedic consultation |

---

## Model Information

### Trained Models (from NB3)

Each binary task trained on:
- **Architectures**: ConvNeXt-L, EfficientNet-B0, DenseNet201, Xception, Swin-T
- **Best Performer**: ConvNeXt-L (seed 42)
- **Training Data**: Kaggle Knee Osteoarthritis Dataset
- **Seeds**: 3 random seeds for robustness (42, 123, 456)

### Performance Metrics

| Task | Threshold | Sensitivity | Specificity | AUC | Accuracy |
|---|---|---|---|---|---|
| Screening (KL≥1) | 0.35 | 0.95 | 0.78 | 0.952 | 0.878 |
| OA (KL≥2) | 0.45 | 0.92 | 0.88 | 0.990 | 0.961 |
| Severe (KL≥3) | 0.50 | 0.94 | 0.89 | 0.990 | 0.961 |

---

## File Structure

```
case study 2/
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   ├── pipeline.py         # Cascade pipeline (Stage 1 + 2)
│   │   ├── agent.py            # AI agent reasoning
│   │   └── weights/            # Model weights (load from NB3)
│   ├── inference/
│   │   ├── preprocess.py       # Image preprocessing
│   │   ├── grad_cam.py         # Heatmap generation
│   │   └── thresholds.py       # Clinical thresholds
│   └── report/
│       └── generator.py        # PDF report generation
│
├── frontend/
│   ├── package.json            # npm dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── index.html              # HTML entry point
│   └── src/
│       ├── main.jsx            # React entry point
│       ├── App.jsx             # Main component
│       ├── index.css           # Global styles
│       └── components/
│           ├── ImageUploader.jsx      # File upload interface
│           ├── ResultsDisplay.jsx     # Results container
│           ├── TrafficLight.jsx       # Risk indicator
│           ├── Visualizer.jsx        # Grad-CAM display
│           └── AgentReport.jsx       # Clinical summary
│
├── Notebooks/
│   ├── NB1_Data_Exploration.ipynb
│   ├── NB2_FiveClass_Benchmark.ipynb
│   ├── NB3_Binary_Classification.ipynb    # ← Model source
│   ├── NB4_CleanLab.ipynb                 # ← Noise analysis
│   ├── NB5_Interpretability.ipynb         # ← Grad-CAM source
│   ├── NB6_External_Validation.ipynb
│   └── NB7_Cascade_Pipeline.ipynb         # ← Cascade logic
│
└── README.md
```

---

## Deployment Options

### Local Development
```bash
# Terminal 1: Backend
cd backend && python main.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

### Docker (Production)
```bash
# Build backend image
cd backend && docker build -t koa-backend .

# Build frontend image
cd frontend && docker build -t koa-frontend .

# Run containers
docker run -p 8000:8000 koa-backend
docker run -p 5173:5173 koa-frontend
```

### Cloud Deployment (Azure/AWS)
- Backend: App Service / EC2
- Frontend: Static Web App / S3 + CloudFront
- Models: Load from Azure Blob Storage / S3

---

## API Reference

### POST /api/predict

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@knee_xray.jpg"
```

**Response:**
```json
{
  "stage_1": {
    "probability": 0.72,
    "label": "Potential OA Detected",
    "threshold_used": 0.35,
    "uncertainty_level": "moderate"
  },
  "stage_2": {
    "probability": 0.68,
    "label": "Moderate OA (KL-2) - Conservative Treatment",
    "confidence": 0.32
  },
  "grad_cam": {
    "heatmap_base64": "data:image/png;base64,...",
    "attention_regions": [
      {"region": "Medial compartment", "intensity": 0.82},
      {"region":"Joint space", "intensity": 0.65}
    ]
  },
  "agent_disclaimer": "...",
  "clinical_summary": "...",
  "traffic_light": "yellow"
}
```

### POST /api/generate-report

**Request:**
```bash
curl -X POST http://localhost:8000/api/generate-report \
  -F "file=@knee_xray.jpg" \
  -F "patient_id=PT-001" \
  -F "include_grad_cam=true" \
  --output report.pdf
```

**Response:** PDF file (binary)

---

## Testing

### Backend Unit Tests
```bash
cd backend
pytest tests/
```

### Frontend Component Tests
```bash
cd frontend
npm run test
```

### Integration Test (E2E)
```bash
# Both services running
./tests/e2e_test.sh
```

---

## Troubleshooting

### "Connection refused" on port 8000
- Ensure backend is running: `python main.py`
- Check firewall settings

### "Module not found" errors
- Install dependencies: `pip install -r requirements.txt` (backend) or `npm install` (frontend)
- Activate virtual environment

### GPU not detected
- Install CUDA: https://developer.nvidia.com/cuda-downloads
- Install cuDNN: https://developer.nvidia.com/cudnn
- Reinstall PyTorch with CUDA support

### Model weights not loading
- Download from NB3 output directory
- Place in `backend/models/weights/`
- Update path in `pipeline.py`

---

## References

- **NB3**: Binary classification model training and optimization
- **NB5**: Grad-CAM interpretability analysis
- **NB7**: Cascade pipeline validation and comparison
- **NB4**: CleanLab noise analysis (KL-1 labeling issues)
- **NB6**: External validation on Mendeley dataset

---

## Citation

If you use this system, please cite:

```
@misc{koa_cascade_2024,
  title={Beyond 5-Class Limitations: A Cascaded AI Architecture for Robust Knee Osteoarthritis Grading},
  author={AI Lab},
  year={2024}
}
```

---

## License

MIT License — See LICENSE file

---

**Last Updated**: March 5, 2024
**Status**: Production Ready ✓
