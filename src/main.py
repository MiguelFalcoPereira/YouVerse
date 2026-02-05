from __future__ import annotations

import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool


from core.config import settings
from services.model import ModelService
from services.preprocessing import preprocess_image_bytes, ImagePreprocessError
from services.postprocessing import top_k_predictions

app = FastAPI(title="ONNX Image Inference Service")


@app.on_event("startup")
def startup() -> None:
    """
    Load model + labels once at startup.
    Store in app.state to avoid globals.
    """
    app.state.model = ModelService(
        model_path=settings.model_path,
        labels_path=settings.labels_path,
    )


@app.post("/infer", tags=["Inference"])
async def infer(file: UploadFile = File(...)) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    model: ModelService = app.state.model

    start = time.perf_counter()
    try:
        input_tensor = await run_in_threadpool(preprocess_image_bytes, image_bytes)
        logits = await run_in_threadpool(model.predict, input_tensor)
        preds = await run_in_threadpool(top_k_predictions, logits, model.labels, settings.top_k)
    except ImagePreprocessError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference failed.") from e

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "top_k": settings.top_k,
        "inference_time_ms": round(elapsed_ms, 2),
        "predictions": [{"label": p.label, "score": p.score} for p in preds],
    }

@app.get("/health", tags=["Health"])
def health() -> dict:
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "ok", "model_loaded": model_loaded}
