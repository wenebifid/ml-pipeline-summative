# src/retrain_utils.py
import os, json
from src.preprocessing import create_generators
from src.model import train_model
from datetime import datetime

METRICS_PATH = "models/metrics.json"

def retrain_from_data(train_dir, val_dir, out_model="models/model_latest.h5", epochs=6, fine_tune_after=None):
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    train_gen, val_gen = create_generators(train_dir, val_dir, batch_size=16)
    model, history = train_model(train_gen, val_gen, out_path=out_model, epochs=epochs, fine_tune_after=fine_tune_after)
    metrics = {
        "train_loss": history.history.get("loss", [])[-1] if history.history.get("loss") else None,
        "val_loss": history.history.get("val_loss", [])[-1] if history.history.get("val_loss") else None,
        "train_acc": history.history.get("accuracy", [])[-1] if history.history.get("accuracy") else None,
        "val_acc": history.history.get("val_accuracy", [])[-1] if history.history.get("val_accuracy") else None,
        "last_trained": datetime.utcnow().isoformat() + "Z"
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
