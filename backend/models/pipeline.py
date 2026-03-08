"""
Binary Cascade Pipeline for Knee OA.

Loads the available ConvNeXt-L binary checkpoints and maps stage outputs
to KL-0..KL-4 probabilities so the existing API contract stays unchanged.
"""

import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TIMM_HOME'] = '/tmp/timm'

import torch
import timm
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# KL Grade mapping
KL_CLASSES = {
    0: "KL-0: Normal",
    1: "KL-1: Doubtful",
    2: "KL-2: Mild OA",
    3: "KL-3: Moderate OA",
    4: "KL-4: Severe OA"
}

class CascadePipeline:
    """Hierarchical binary cascade using ConvNeXt-L checkpoints."""

    def __init__(
        self,
        model_dir: str,
        device: str,
        screening_model_path: str,
        severe_model_path: str,
        oa_model_path: Optional[str] = None,
        koa_model_path: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.is_loaded = False

        self.arch_name = "ConvNeXt-L"
        self.model_name = "convnext_large"
        self.seed = 42

        # KL priors used only when OA model is unavailable.
        self.kl12_split = (0.41, 0.59)  # KL-1 vs KL-2 within non-severe OA
        self.kl34_split = (0.81, 0.19)  # KL-3 vs KL-4 within severe OA

        self.screening_model_path = screening_model_path
        self.severe_model_path = severe_model_path
        self.oa_model_path = oa_model_path

        self.screening_model = None
        self.severe_model = None
        self.oa_model = None

        # Backward compatibility for Grad-CAM caller expecting `cascade.model`.
        self.model = None

        # `koa_model_path` is retained for compatibility with existing call sites.
        _ = koa_model_path
        self._load_models()

    def _create_binary_model(self):
        return timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=2,
        ).to(self.device)

    @staticmethod
    def _unwrap_state_dict(raw_obj):
        if isinstance(raw_obj, dict) and "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
            return raw_obj["state_dict"]
        return raw_obj

    def _load_checkpoint(self, model: torch.nn.Module, ckpt_path: str, label: str):
        path = Path(ckpt_path)
        print(f"  [load] {label}: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Required model not found: {path}")

        raw_obj = torch.load(str(path), map_location=self.device)
        state_dict = self._unwrap_state_dict(raw_obj)
        model.load_state_dict(state_dict)
        model.eval()

    def _load_models(self):
        try:
            print("  [1] Creating ConvNeXt-L binary models...")
            self.screening_model = self._create_binary_model()
            self.severe_model = self._create_binary_model()

            if self.oa_model_path:
                self.oa_model = self._create_binary_model()

            print("  [2] Loading cascade checkpoints...")
            self._load_checkpoint(self.screening_model, self.screening_model_path, "Screening (KL>=1)")
            self._load_checkpoint(self.severe_model, self.severe_model_path, "Severe (KL>=3)")

            if self.oa_model is not None:
                self._load_checkpoint(self.oa_model, self.oa_model_path, "OA (KL>=2)")
                print("  [3] OA checkpoint loaded.")
            else:
                print("  [3] OA checkpoint not configured. Using prior-based KL-1/KL-2 split.")

            # Keep Grad-CAM working with existing call sites.
            self.model = self.screening_model

            self.is_loaded = True
            print("[Pipeline] ✓ Binary cascade loaded successfully")

        except Exception as e:
            import traceback
            print(f"[Pipeline] ERROR: {e}")
            traceback.print_exc()
            raise
    
    @torch.no_grad()
    def forward(self, image_tensor: torch.Tensor) -> Dict:
        """
        Forward pass through binary cascade and projection to KL-0..KL-4.

        Output schema matches the previous 5-class API contract.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        screen_logits = self.screening_model(image_tensor)
        severe_logits = self.severe_model(image_tensor)

        p_ge1 = float(torch.softmax(screen_logits, dim=1)[0, 1].cpu().item())
        p_ge3 = float(torch.softmax(severe_logits, dim=1)[0, 1].cpu().item())

        # Enforce monotonicity P(KL>=3) <= P(KL>=1)
        p_ge3 = min(max(p_ge3, 0.0), max(p_ge1, 0.0))
        p0 = max(0.0, 1.0 - p_ge1)

        if self.oa_model is not None:
            oa_logits = self.oa_model(image_tensor)
            p_ge2 = float(torch.softmax(oa_logits, dim=1)[0, 1].cpu().item())
            p_ge2 = min(max(p_ge2, p_ge3), p_ge1)

            p1 = max(0.0, p_ge1 - p_ge2)
            p2 = max(0.0, p_ge2 - p_ge3)
        else:
            p_non_severe_oa = max(0.0, p_ge1 - p_ge3)
            p1 = p_non_severe_oa * self.kl12_split[0]
            p2 = p_non_severe_oa * self.kl12_split[1]

        p3 = p_ge3 * self.kl34_split[0]
        p4 = p_ge3 * self.kl34_split[1]

        probs = np.array([p0, p1, p2, p3, p4], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0:
            probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            probs = probs / total

        pred_class = int(np.argmax(probs))
        kl_confidence = float(probs[pred_class])

        # Report confidence as decision-path confidence for the active cascade.
        # This avoids artificially low confidence for KL-2 when KL-1/KL-2 are
        # reconstructed from priors due to a missing OA checkpoint.
        if pred_class == 0:
            path_confidence = p0
        elif pred_class in (1, 2):
            if self.oa_model is not None:
                path_confidence = p1 if pred_class == 1 else p2
            else:
                path_confidence = max(0.0, p_ge1 - p_ge3)
        else:  # KL-3 / KL-4
            path_confidence = p_ge3

        confidence = float(max(0.0, min(1.0, path_confidence)))
        all_predictions = {f"KL-{i}": float(probs[i]) for i in range(5)}

        return {
            # Pseudo-logits retained for compatibility with debug views.
            "logits": np.log(np.clip(probs, 1e-12, 1.0)).tolist(),
            "probabilities": probs.tolist(),
            "predicted_class": pred_class,
            "predicted_label": KL_CLASSES[pred_class],
            "confidence": confidence,
            "kl_confidence": kl_confidence,
            "all_predictions": all_predictions,
            "stage_probs": {
                "screening_kl_ge_1": float(p_ge1),
                "severe_kl_ge_3": float(p_ge3),
            },
        }
    
    def get_model_info(self):
        """Return metadata about the models"""
        return {
            "architecture": self.arch_name,
            "model_type": "binary cascade projected to KL-0..KL-4",
            "num_classes": 5,
            "seed": self.seed,
            "classes": KL_CLASSES,
            "screening_model_path": self.screening_model_path,
            "severe_model_path": self.severe_model_path,
            "oa_model_path": self.oa_model_path,
        }
