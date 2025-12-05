# src/prediction.py
import os
from tensorflow.keras.models import load_model
from src.preprocessing import load_image_for_prediction

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_latest.h5")

def load_trained_model(model_path=MODEL_PATH):
    model = load_model(model_path)
    return model

def predict_image(model, image_path, class_indices):
    img_arr = load_image_for_prediction(image_path)
    preds = model.predict(img_arr)
    top_idx = preds[0].argsort()[-3:][::-1]
    inv_map = {v:k for k,v in class_indices.items()}
    results = []
    for idx in top_idx:
        results.append({
            "class": inv_map.get(int(idx), f"class_{idx}"),
            "probability": float(preds[0][idx])
        })
    return results
