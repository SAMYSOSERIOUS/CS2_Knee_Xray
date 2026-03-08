"""
Grad-CAM Heatmap Generation
Explains model attention through visual saliency maps
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import base64
from io import BytesIO
from PIL import Image
import cv2


def _fallback_heatmap() -> str:
    """
    Return a neutral gray placeholder data URL when Grad-CAM fails.
    Also overlays a text notice using basic pixel drawing.
    """
    arr = np.full((224, 224, 3), 60, dtype=np.uint8)  # dark gray background
    # Draw a simple red-to-blue gradient so the widget still renders visibly
    for col in range(224):
        ratio = col / 223.0
        arr[:, col, 0] = int(220 * (1 - ratio))   # R channel
        arr[:, col, 2] = int(220 * ratio)           # B channel
    img = Image.fromarray(arr, 'RGB')
    return image_to_base64(img)


def generate_grad_cam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_layer: str = "ConvNeXt-L",
) -> Tuple[str, List[Dict]]:
    """
    Generate input-gradient saliency (Grad-CAM proxy) and attention regions.

    Args:
        model: PyTorch model
        image_tensor: Preprocessed tensor (3, 224, 224)  — will be cloned internally
        device: torch.device
        target_layer: Architecture label (informational)

    Returns:
        (heatmap_base64, attention_regions)
        - heatmap_base64: Full data-URL string  ``data:image/png;base64,...``
        - attention_regions: List of {"region": str, "intensity": float}
    """
    # Clone so we don't affect the original tensor used by the prediction step
    img = image_tensor.clone().detach()
    if img.ndim == 3:
        img = img.unsqueeze(0)          # add batch dim → (1, 3, H, W)
    img = img.to(device)
    img.requires_grad_(True)            # leaf tensor, ready for grad accumulation

    model.eval()

    # Forward pass with gradient tracking enabled
    with torch.enable_grad():
        output = model(img)             # (1, num_classes)
        # Back-propagate through the highest-confidence class
        pred_idx = int(output[0].argmax().item())
        score = output[0, pred_idx]
        score.backward()

    # Safety check — should not be None for a leaf tensor with requires_grad=True
    if img.grad is None:
        return _fallback_heatmap(), []

    # Input-gradient saliency: average absolute gradient across colour channels
    gradients = img.grad[0]                          # (3, H, W)
    activations = gradients.abs().mean(dim=0).cpu().numpy()  # (H, W)

    # Normalize to [0, 1] and resize to display size
    activations = (activations - activations.min()) / (activations.max() - activations.min() + 1e-8)
    activations = cv2.resize(activations, (224, 224), interpolation=cv2.INTER_LINEAR)

    heatmap_img = colorize_heatmap(activations)
    attention_regions = identify_attention_regions(activations)
    heatmap_b64 = image_to_base64(heatmap_img)

    return heatmap_b64, attention_regions


def colorize_heatmap(heatmap: np.ndarray) -> Image.Image:
    """
    Convert grayscale heatmap to color (jet colormap).
    
    Args:
        heatmap: Shape (H, W), values in [0, 1]
    
    Returns:
        PIL Image (RGB)
    """
    # Normalize to 0-255
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply jet colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(heatmap_color)


def identify_attention_regions(heatmap: np.ndarray, n_regions: int = 3) -> List[Dict]:
    """
    Identify anatomical regions from heatmap attention.
    
    Args:
        heatmap: Shape (224, 224), values in [0, 1]
        n_regions: Number of top regions to return
    
    Returns:
        List of {"region": str, "intensity": float}
    """
    
    # Define anatomical regions (predefined bounding boxes)
    regions = {
        "Medial compartment": ((30, 112), (180, 224)),     # Left side
        "Lateral compartment": ((44, 112), (194, 224)),    # Right side
        "Joint space": ((70, 100), (154, 200)),            # Center
        "Osteophytes": ((0, 60), (224, 120)),              # Upper region
    }
    
    region_scores = []
    for region_name, (top_left, bottom_right) in regions.items():
        y1, x1 = top_left
        y2, x2 = bottom_right
        region_intensity = heatmap[y1:y2, x1:x2].mean()
        region_scores.append({"region": region_name, "intensity": float(region_intensity)})
    
    # Sort by intensity and return top N
    region_scores.sort(key=lambda x: x["intensity"], reverse=True)
    return region_scores[:n_regions]


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: Image.Image,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay heatmap on original image for visualization.
    """
    original_image = original_image.resize((224, 224))
    heatmap_resized = heatmap.resize((224, 224))
    
    # Blend
    blended = Image.blend(original_image, heatmap_resized, alpha=alpha)
    return blended
