"""
Hierarchical Clinical Decision Tree for Knee OA Detection
FastAPI Backend Service
"""

import os
import torch
import timm
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image
import json

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn

# Local imports
from models.pipeline import CascadePipeline
from models.agent import ClinicalAgent
from inference.preprocess import preprocess_image
from inference.grad_cam import generate_grad_cam, _fallback_heatmap
from inference.thresholds import CLINICAL_THRESHOLDS
from report.generator import generate_pdf_report
import data_manager
import base64

# ============================================================================
# INITIALIZATION
# ============================================================================

app = FastAPI(
    title="KOA Clinical Decision Support",
    description="Hierarchical cascade pipeline for knee OA diagnosis",
    version="1.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[BACKEND] Device: {DEVICE}")

# Model paths
MODEL_DIR = Path(__file__).parent / "models" / "weights"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# KOA folder with trained cascade models
# In Docker the SAVED_MODELS_DIR env var is set to /saved_models (mounted volume)
_default_saved_models = Path("c:/case study 2/KOA/saved_models")
SAVED_MODELS_DIR = Path(os.environ.get("SAVED_MODELS_DIR", str(_default_saved_models)))

SCREENING_MODEL_PATH = SAVED_MODELS_DIR / "ConvNeXt-L_Screening_seed42.pth"
SEVERE_MODEL_PATH = SAVED_MODELS_DIR / "ConvNeXt-L_Severe_seed42.pth"
# Optional third-stage checkpoint if later added:
OA_MODEL_PATH = SAVED_MODELS_DIR / "ConvNeXt-L_OA_seed42.pth"

# Initialize models
print("[BACKEND] Loading binary cascade pipeline (Screening + Severe)...")
cascade = CascadePipeline(
    model_dir=str(MODEL_DIR),
    device=DEVICE,
    screening_model_path=str(SCREENING_MODEL_PATH),
    severe_model_path=str(SEVERE_MODEL_PATH),
    oa_model_path=str(OA_MODEL_PATH) if OA_MODEL_PATH.exists() else None,
)

print("[BACKEND] Loading clinical agent...")
agent = ClinicalAgent()

print("[BACKEND] Initializing dataset manager...")
custom_kaggle_creds = Path(os.environ.get("KAGGLE_CONFIG_PATH", "c:/case study 2/Secrets/kaggle.json"))
data_manager.setup_credentials(custom_kaggle_creds)

print("[BACKEND] ✓ System initialized")


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": cascade.is_loaded,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload X-ray image → returns 5-class KL grade prediction
    
    Response:
    {
        "kl_grade": int,
        "kl_label": str,
        "confidence": float,
        "all_probabilities": {"KL-0": float, ..., "KL-4": float},
        "traffic_light": "green" | "yellow" | "red",
        "recommendation": str,
        "grad_cam": {
            "heatmap_base64": str,
            "attention_regions": list
        },
        "clinical_summary": str
    }
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Convert to tensor
        tensor_img = preprocess_image(image)
        
        # Run 5-class prediction
        prediction = cascade.forward(tensor_img)
        
        kl_grade = prediction["predicted_class"]
        kl_label = prediction["predicted_label"]
        confidence = prediction["confidence"]
        all_probs = prediction["all_predictions"]
        
        # Map KL grade to traffic light
        if kl_grade <= 1:  # KL-0, KL-1
            traffic_light = "green"
            risk_level = "Low Risk"
            recommendation = "Standard preventive care, follow-up in 2-3 years"
        elif kl_grade == 2:  # KL-2 Mild
            traffic_light = "yellow"
            risk_level = "Moderate OA"
            recommendation = "Conservative treatment (exercise, NSAIDs), follow-up in 12-24 months"
        else:  # KL-3, KL-4
            traffic_light = "red"
            risk_level = "Severe OA"
            recommendation = "Orthopedic consultation for surgical evaluation"
        
        # Generate Grad-CAM heatmap
        try:
            heatmap_b64, attention_regions = generate_grad_cam(
                cascade.model,
                tensor_img,
                device=DEVICE
            )
        except Exception as cam_err:
            print(f"[Grad-CAM] Warning — saliency failed: {cam_err}")
            heatmap_b64 = _fallback_heatmap()
            attention_regions = []
        
        # Generate clinical summary
        clinical_summary = f"""## {risk_level}
**Predicted:** {kl_label}
**Confidence:** {confidence*100:.1f}%
**Recommendation:** {recommendation}"""
        
        return {
            "kl_grade": kl_grade,
            "kl_label": kl_label,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "traffic_light": traffic_light,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "grad_cam": {
                "heatmap_base64": heatmap_b64,
                "attention_regions": attention_regions,
            },
            "clinical_summary": clinical_summary,
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ReportRequest(BaseModel):
    patient_id: str = "UNKNOWN"
    include_grad_cam: bool = True
    kl_grade: Optional[int] = None
    kl_label: Optional[str] = None
    confidence: Optional[float] = None
    all_probabilities: Optional[Dict[str, float]] = None
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None
    traffic_light: Optional[str] = None


def _derive_stage_probs(payload: ReportRequest) -> tuple[float, Optional[float]]:
    """Derive stage probabilities from the frontend prediction payload."""
    probs = payload.all_probabilities or {}

    # Stage 1 approximates P(KL>=1); use KL-0 probability when available.
    if "KL-0" in probs:
        stage1_prob = 1.0 - float(probs.get("KL-0", 0.0))
    elif payload.kl_grade is not None:
        stage1_prob = 0.9 if payload.kl_grade >= 1 else 0.1
    elif payload.confidence is not None:
        stage1_prob = float(payload.confidence)
    else:
        stage1_prob = 0.5

    # Stage 2 approximates P(KL>=3); sum KL-3 and KL-4 when available.
    if "KL-3" in probs or "KL-4" in probs:
        stage2_prob = float(probs.get("KL-3", 0.0)) + float(probs.get("KL-4", 0.0))
    elif payload.kl_grade is not None:
        stage2_prob = 0.9 if payload.kl_grade >= 3 else 0.1
    else:
        stage2_prob = None

    stage1_prob = max(0.0, min(1.0, stage1_prob))
    if stage2_prob is not None:
        stage2_prob = max(0.0, min(1.0, stage2_prob))

    return stage1_prob, stage2_prob

@app.post("/api/generate-report")
async def generate_report(payload: ReportRequest):
    """
    Generate PDF clinical report
    """
    try:
        stage1_prob, stage2_prob = _derive_stage_probs(payload)
        patient_id = payload.patient_id
        include_grad_cam = payload.include_grad_cam
        contents = b""
        
        # Generate report
        pdf_bytes = generate_pdf_report(
            patient_id=patient_id,
            image_bytes=contents,
            stage1_prob=float(stage1_prob),
            stage2_prob=float(stage2_prob) if stage2_prob is not None else None,
            include_grad_cam=include_grad_cam,
            kl_grade=payload.kl_grade,
            kl_label=payload.kl_label,
            confidence=payload.confidence,
            all_probabilities=payload.all_probabilities,
            risk_level=payload.risk_level,
            recommendation=payload.recommendation,
            traffic_light=payload.traffic_light,
        )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="KOA_Report_{patient_id}.pdf"'
            },
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/thresholds")
def get_thresholds():
    """Return current clinical thresholds"""
    return CLINICAL_THRESHOLDS


@app.get("/api/info")
def get_info():
    """Return system information"""
    return {
        "version": "1.0.0",
        "pipeline": "Binary Cascade (Screening KL>=1 + Severe KL>=3)",
        "architecture": "ConvNeXt-L (binary)",
        "device": str(DEVICE),
        "stage1_task": "KL≥1 Screening (Healthy/Subtle vs Definite OA)",
        "stage2_task": "KL≥3 Severity (Moderate vs Severe OA)",
        "cascade_4class_acc": 0.878,
        "note": "KL-0..KL-4 probabilities are projected from binary stage outputs",
    }


# ============================================================================
# IMAGE BROWSING API (KAGGLE DATASET)
# ============================================================================

@app.get("/api/available-images")
def get_available_images(split: str = "test", limit: int = 50):
    """
    Get list of available images from Kaggle dataset (test/val splits only)
    
    PREVENTS DATA LEAKAGE: Training images are NOT accessible through this endpoint
    
    Query Parameters:
        split: "test" or "val" (default: "test")
        limit: Maximum number of images to return (default: 50)
    
    Returns:
        List of image metadata with fields:
        - filename: Image filename
        - kl_grade: KL grade 0-4
        - split: Data split
        - file_size: File size in bytes
    """
    try:
        images = data_manager.list_available_images(split=split, limit=limit)
        
        return {
            "split": split,
            "count": len(images),
            "images": images,
            "safe_to_use": split.lower() in ["test", "val"],
            "note": "Training images are excluded to prevent data leakage"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset-stats")
def get_dataset_stats(split: str = "test"):
    """
    Get statistics about the available dataset
    
    Returns:
        Statistics including:
        - total_images: Total number of images
        - grade_distribution: Count by KL grade
        - total_size_mb: Total dataset size
        - safe_to_use: Whether split is safe for evaluation
    """
    try:
        stats = data_manager.get_statistics(split=split)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-from-dataset")
async def predict_from_dataset(filename: str, split: str = "test"):
    """
    Run prediction on an image from the Kaggle dataset
    
    Prevents data leakage by only allowing test/val splits
    
    Request Body:
        filename: Image filename from the dataset
        split: Data split ("test" or "val")
    
    Returns:
        Same as /api/predict - full prediction with Grad-CAM
    """
    try:
        # Safety check: prevent training data access
        if split.lower() == "train":
            raise HTTPException(
                status_code=403,
                detail="⛔ DATA LEAKAGE PREVENTION: Training data is not accessible for prediction"
            )
        
        # Get image metadata
        image_dict = data_manager.get_image_by_filename(filename, split=split)
        if not image_dict:
            raise HTTPException(
                status_code=404,
                detail=f"Image '{filename}' not found in '{split}' split"
            )
        
        # Load image file
        image_bytes = data_manager.load_image_bytes(image_dict)
        if not image_bytes:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load image file: {filename}"
            )
        
        # Preprocess and predict
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        tensor_img = preprocess_image(image)
        prediction = cascade.forward(tensor_img)
        
        kl_grade = prediction["predicted_class"]
        kl_label = prediction["predicted_label"]
        confidence = prediction["confidence"]
        all_probs = prediction["all_predictions"]
        
        # Map KL grade to traffic light
        if kl_grade <= 1:
            traffic_light = "green"
            risk_level = "Low Risk"
            recommendation = "Standard preventive care, follow-up in 2-3 years"
        elif kl_grade == 2:
            traffic_light = "yellow"
            risk_level = "Moderate OA"
            recommendation = "Conservative treatment (exercise, NSAIDs), follow-up in 12-24 months"
        else:
            traffic_light = "red"
            risk_level = "Severe OA"
            recommendation = "Orthopedic consultation for surgical evaluation"
        
        # Generate Grad-CAM
        try:
            heatmap_b64, attention_regions = generate_grad_cam(
                cascade.model, tensor_img, device=DEVICE
            )
        except Exception as cam_err:
            print(f"[Grad-CAM] Warning — saliency failed: {cam_err}")
            heatmap_b64 = _fallback_heatmap()
            attention_regions = []
        
        # Clinical summary
        clinical_summary = f"""## {risk_level}
**Predicted:** {kl_label}
**Confidence:** {confidence*100:.1f}%
**Source:** Kaggle Dataset ({split} split)
**Recommendation:** {recommendation}"""
        
        return {
            "filename": filename,
            "split": split,
            "kl_grade": kl_grade,
            "kl_label": kl_label,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "traffic_light": traffic_light,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "grad_cam": {
                "heatmap_base64": heatmap_b64,
                "attention_regions": attention_regions,
            },
            "clinical_summary": clinical_summary,
            "is_from_dataset": True,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image-thumbnail")
def get_image_thumbnail(kaggle_path: str):
    """
    Return a real knee X-ray image as base64 PNG, streamed from Kaggle (no disk I/O).

    Query Parameters:
        kaggle_path: The Kaggle-relative path, e.g. "test/0/9003175L.png"

    Returns:
        { "image_base64": "<base64 PNG>", "kaggle_path": "...", "media_type": "image/png" }
    """
    try:
        # Prevent access to training data
        if kaggle_path.startswith("train/"):
            raise HTTPException(
                status_code=403,
                detail="⛔ DATA LEAKAGE PREVENTION: Training images are not accessible"
            )

        img_dict = {"kaggle_path": kaggle_path}
        image_bytes = data_manager.load_image_bytes(img_dict)
        if not image_bytes:
            raise HTTPException(status_code=404, detail=f"Image not found: {kaggle_path}")

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "image_base64": encoded,
            "kaggle_path": kaggle_path,
            "media_type": "image/png",
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
