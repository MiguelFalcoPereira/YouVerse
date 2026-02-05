import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def _set_required_env_vars_for_tests():
    os.environ.setdefault("MODEL_PATH", "src/models/resnet50-v2-7.onnx")
    os.environ.setdefault("LABELS_PATH", "src/models/imagenet_classes.txt")
    os.environ.setdefault("TOP_K", "3")
    os.environ.setdefault("NUM_THREADS", "1")
