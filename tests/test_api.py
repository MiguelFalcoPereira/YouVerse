import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient


def make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeModelService:
    def __init__(self, *args, **kwargs):
        # 3 fake labels; in real model it's 1000
        self.labels = ["class0", "class1", "class2"]

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        # Return logits shaped (1, 3)
        return np.array([[0.1, 0.2, 5.0]], dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    # Import inside fixture so monkeypatch applies before app startup runs
    import main

    # Patch the ModelService used in main.startup()
    monkeypatch.setattr(main, "ModelService", FakeModelService)

    # Patch preprocessing to return a dummy tensor the model accepts
    monkeypatch.setattr(
        main,
        "preprocess_image_bytes",
        lambda b: np.zeros((1, 3, 224, 224), dtype=np.float32),
    )

    # Patch postprocessing to avoid relying on 1000 labels
    monkeypatch.setattr(
        main,
        "top_k_predictions",
        lambda logits, labels, k: [
            type("P", (), {"label": "class2", "score": 0.9})(),
            type("P", (), {"label": "class1", "score": 0.05})(),
            type("P", (), {"label": "class0", "score": 0.05})(),
        ][:k],
    )

    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_infer_returns_predictions(client):
    image_bytes = make_test_image_bytes()

    files = {"file": ("test.png", image_bytes, "image/png")}
    r = client.post("/infer", files=files)

    assert r.status_code == 200
    data = r.json()

    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 3

    first = data["predictions"][0]
    assert set(first.keys()) == {"label", "score"}
    assert first["label"] == "class2"
    assert isinstance(first["score"], float)
