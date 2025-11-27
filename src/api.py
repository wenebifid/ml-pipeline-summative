# src/api.py
import os
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, start_http_server
from src.prediction import load_trained_model, predict_image
from src.preprocessing import create_generators
from src.model import train_model
import shutil

REQUEST_COUNT = Counter('api_request_count', 'Total HTTP requests')
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Latency of HTTP requests')

app = FastAPI(title="EuroSAT Image Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_latest.h5")
DATA_DIR = os.environ.get("DATA_DIR", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")

model = None
class_indices = {}

@app.on_event("startup")
def startup_event():
    global model, class_indices
    start_http_server(8001)
    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
    # infer classes from train dir if exists
    if os.path.exists(TRAIN_DIR):
        subdirs = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
        class_indices.update({name: idx for idx, name in enumerate(subdirs)})

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        suffix = os.path.splitext(file.filename)[1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file.file.read())
        tmp.flush()
        tmp.close()
        if model is None:
            return JSONResponse({"error":"No model available"}, status_code=503)
        res = predict_image(model, tmp.name, class_indices)
        os.unlink(tmp.name)
        REQUEST_LATENCY.observe(time.time() - start)
        return {"predictions": res}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/trigger-retrain")
def trigger_retrain(epochs: int = Form(5)):
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        return JSONResponse({"error":"train/test directories missing"}, status_code=400)
    start_time = time.time()
    global model
    try:
        train_gen, val_gen = create_generators(TRAIN_DIR, VAL_DIR, batch_size=32)
        model, history = train_model(train_gen, val_gen, out_path=MODEL_PATH, epochs=epochs)
        duration = time.time() - start_time
        return {"status":"retrained", "duration_seconds": duration}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000)
