from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from core.config import settings

class ModelService:
    def __init__(self, model_path: Path, labels_path: Path):
        self.model_path = model_path
        self.labels_path = labels_path

        self.session: ort.InferenceSession | None = None
        self.labels: List[str] = []

        self._load_labels()
        self._load_model()

    def _load_labels(self) -> None:
        with self.labels_path.open("r") as f:
            self.labels = [line.strip() for line in f if line.strip()]

        if not self.labels:
            raise RuntimeError("Labels file is empty")

    def _load_model(self) -> None:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = settings.num_threads

        self.session = ort.InferenceSession(
            self.model_path.as_posix(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("Model session is not initialized")

        inputs = {self.session.get_inputs()[0].name: input_tensor}
        outputs = self.session.run(None, inputs)

        return outputs[0]

    def get_label(self, index: int) -> str:
        try:
            return self.labels[index]
        except IndexError:
            raise ValueError(f"Invalid class index: {index}")