# ONNX Image Inference Service
FastAPI-based backend service that performs image classification inference using a provided ONNX model.

## Setup and Run Instructions

### Prerequisites:
- Python 3.12+
- pip

### Setup
1. Create a Virtual Environment
```bash
python -m venv .venv
#powershell
.venv\Scripts\Activate.ps1
#bash
source .venv/bin/activate
```
2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```
3. Set Env Variables
```bash
#powershell
$env:PYTHONPATH="src"
$env:MODEL_PATH="src\models\resnet50-v2-7.onnx"
$env:LABELS_PATH="src\models\imagenet_classes.txt"
$env:TOP_K="3"
$env:NUM_THREADS="4"
```
```bash
#bash
export PYTHONPATH=src
export MODEL_PATH=src/models/resnet50-v2-7.onnx
export LABELS_PATH=src/models/imagenet_classes.txt
export TOP_K=3
export NUM_THREADS=4
```
4. Run the API Server
```bash
#powershell
python -m uvicorn main:app
#bash
py -m uvicorn main:app
```
5. Run Tests
```bash
pytest -q
```

## Api Usage
When runing the api server, the service will be available at http://127.0.0.1:8000.
Use Swagger UI at http://127.0.0.1:8000/docs and test both endpoints.
- Possible response for /infer endpoint:
```json
{
  "top_k": 3,
  "inference_time_ms": 64.7,
  "predictions": [
    {
      "label": "Siamese cat",
      "score": 0.9762075543403625
    },
    {
      "label": "Egyptian cat",
      "score": 0.006209633778780699
    },
    {
      "label": "lynx",
      "score": 0.0014239016454666853
    }
  ]
}
```
- Possible response for /health enpoint:
```json
{
  "status": "ok",
  "model_loaded": true
}
```
## Key Design Decisions and Tradeoffs

### Model loaded at application startup
The ONNX model and labels are loaded once during application startup and stored in the application state.
This reduces latency per request and if the application will fail at startup if the model is missing or invalid,
instead of starting in a degraded state.

### Clean Separation of Concerns
The codebase is organized to separate responsibilities across distinct modules:

- The API layer (main.py) is responsible for HTTP request handling, validation, and response formatting;

- Image preprocessing is isolated in a dedicated module;

- Model inference is encapsulated in a service that loads the ONNX model and performs its dedicated tasks;

- Postprocessing logic (softmax, top-K selection, label mapping) is also separated as a different service.

This structure keeps the inference pipeline readable and testable. Each component can be tested independently, 
and the overall flow remains easy to follow.


## What could be improved?
### Model lifecycle and flexibility
- Support loading different models via configuration;
- Introduce optional batching for higher throughput workloads;
- Add a model warm-up step during startup to reduce first-request latency.
### Error handling and Observability
- Add loggings;
- Improve error responses.