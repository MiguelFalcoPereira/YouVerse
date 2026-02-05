from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Accepts shape (N, K) or (K,).
    Returns same shape.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=1, keepdims=True)


def top_k_predictions(
    logits: np.ndarray,
    labels: List[str],
    k: int,
) -> List[Prediction]:
    """
    logits: (1, 1000) or (1000,)
    returns: list of Prediction sorted by score desc
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    probs = softmax(logits)  # (1, 1000)
    probs_1d = probs[0]

    if probs_1d.shape[0] != len(labels):
        raise ValueError(
            f"Label count ({len(labels)}) does not match output size ({probs_1d.shape[0]})"
        )

    k = min(k, probs_1d.shape[0])

    # argpartition is faster than full sort for large K
    idx = np.argpartition(-probs_1d, k - 1)[:k]
    idx = idx[np.argsort(-probs_1d[idx])]

    return [Prediction(label=labels[i], score=float(probs_1d[i])) for i in idx]
