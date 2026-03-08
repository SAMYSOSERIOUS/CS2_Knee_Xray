"""
Image Preprocessing Pipeline
Handles DICOM and JPG formats, normalization, and tensor conversion
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from io import BytesIO


# ImageNet normalization (standard for timm models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image to tensor suitable for model inference.
    
    Args:
        image: PIL Image (already converted to RGB)
    
    Returns:
        torch.Tensor: Shape (3, 224, 224)
    """
    
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    tensor = transform(image)
    return tensor


def preprocess_batch(images: list) -> torch.Tensor:
    """
    Preprocess batch of images.
    
    Args:
        images: List of PIL Images
    
    Returns:
        torch.Tensor: Shape (batch_size, 3, 224, 224)
    """
    tensors = [preprocess_image(img) for img in images]
    return torch.stack(tensors)


def postprocess_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize heatmap for visualization.
    
    Args:
        heatmap: Shape (H, W), values in [0, 1]
    
    Returns:
        np.ndarray: Normalized heatmap
    """
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap
