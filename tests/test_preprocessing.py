import io
import numpy as np
from PIL import Image

from services.preprocessing import preprocess_image_bytes


def test_preprocess_image_bytes_returns_expected_tensor():
    img = Image.new("RGB", (300, 150), color=(255, 0, 0))  # red
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    tensor = preprocess_image_bytes(image_bytes)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
