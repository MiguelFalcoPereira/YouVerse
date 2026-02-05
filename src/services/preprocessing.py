from __future__ import annotations

from io import BytesIO
from typing import Final

import numpy as np
from PIL import Image

# ImageNet normalization constants
IMAGENET_MEAN: Final[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: Final[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE: Final[int] = 224


class ImagePreprocessError(ValueError):
    """Raised when an uploaded file cannot be processed as an image."""


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a ResNet-50 input tensor.
    Returns:
        np.ndarray: float32 tensor shaped (1, 3, 224, 224)
    """
    if not image_bytes:
        raise ImagePreprocessError("Empty file")

    # 1) Decode
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")  # ensure 3 channels
            img = img.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.BILINEAR)
    except Exception as e:
        raise ImagePreprocessError("Invalid image file") from e

    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Normalization
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32, copy=False)
