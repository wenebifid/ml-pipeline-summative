# app_streamlit.py
import streamlit as st
import os, tempfile, zipfile, time, json, shutil
from PIL import Image
import numpy as np
import tensorflow as tf
from src.preprocessing import load_image_for_prediction, create_generators
from src.model import train_model
from src.prediction import predict_image, load_trained_model
from src.utils import get_class_indices_from_dir
from src.charts import plot_class_distribution, plot_confusion_matrix_png
from src.s3_utils import download_model_from_s3, upload_model_to_s3

st.set_page_config(page_title="EuroSAT Streamlit Dashboard", layout="wide")
st.title("EuroSAT — Streamlit MLOps Dashboard (ResNet50)")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_latest.h5")
DATA_DIR = os.environ.get("DATA_DIR", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "")

@st.cache_resource
def load_model_cached(path=MODEL_PATH):
    if S3_BUCKET and S3_MODEL_KEY:
        try:
            download_model_from_s3(S3_BUCKET, S3_MODEL_KEY, path)
        except Exception as e:
            st.warning(f"Could not download from S3: {e}")
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        return model
    return None

model = load_model_cached()

st.sidebar.header("Model Controls")
if st.sidebar.button("Reload model"):
    load_model_cached.clear()
    model = load_model_cached()
    st.experimental_rerun()

if model is None:
    st.sidebar.warning("No model found at models/model_latest.h5. Use the Notebook to train or upload a model.")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Retrain", "Insights", "System"])

with tab1:
    st.header("Single Image Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name)
        if st.button("Predict"):
            if model is None:
                st.error("No model available.")
            else:
                if os.path.exists(TRAIN_DIR):
                    class_indices = get_class_indices_from_dir(TRAIN_DIR)
                else:
                    class_indices = {}
                preds = predict_image(model, tmp.name, class_indices)
                st.json(preds)
        tmp.close()

with tab2:
    st.header("Bulk Upload for Retraining")
    st.write("Upload a ZIP file containing `train/<class>/*` and `test/<class>/*` folders or files named `class__imagename.jpg`.")
    zip_file = st.file_uploader("Upload ZIP", type=["zip"])
    if zip_file:
        ztmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        ztmp.write(zip_file.getvalue())
        ztmp.flush()
        ztmp.close()
        extract_to = os.path.join(UPLOADS_DIR, f"upload_{int(time.time())}")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(ztmp.name, 'r') as zf:
            zf.extractall(extract_to)
        st.success(f"Extracted to {extract_to}")
        if st.button("Move to dataset"):
            dest = os.path.join(DATA_DIR, f"staged_{int(time.time())}")
            shutil.copytree(extract_to, dest)
            st.success(f"Staged to {dest}")

    st.markdown("### Trigger Retrain")
    epochs = st.number_input("Epochs", min_value=1, max_value=50, value=6)
    fine_tune_after = st.number_input("Fine-tune freeze first N base layers", min_value=0, value=100, step=1)
    if st.button("Retrain now"):
        if not (os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR)):
            st.error("Please ensure data/train and data/test exist with class subfolders.")
        else:
            with st.spinner("Training... this may take several minutes"):
                train_gen, val_gen = create_generators(TRAIN_DIR, TEST_DIR, batch_size=16)
                model_obj, history = train_model(train_gen, val_gen, out_path=MODEL_PATH, epochs=epochs, fine_tune_after=fine_tune_after)
                st.success("Retraining complete.")
                if S3_BUCKET:
                    try:
                        upload_model_to_s3(S3_BUCKET, f"model_{int(time.time())}.h5", MODEL_PATH)
                        st.info("Uploaded model to S3.")
                    except Exception as e:
                        st.warning(f"Failed to upload to S3: {e}")

with tab3:
    st.header("Insights & Visualizations")
    st.write("Class distribution (training set)")
    if os.path.exists(TRAIN_DIR):
        plot_path = plot_class_distribution(TRAIN_DIR, out_path="models/class_distribution.png")
        st.image(plot_path, caption="Class distribution")
    else:
        st.info("No training data found at data/train")

    st.write("Confusion matrix (if models and test set exist)")
    cm_path = "models/confusion_matrix.png"
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_DIR):
        try:
            plot_confusion_matrix_png(MODEL_PATH, TEST_DIR, out_path=cm_path)
            st.image(cm_path, caption="Confusion matrix")
        except Exception as e:
            st.warning(f"Could not compute confusion matrix: {e}")
    else:
        st.info("Provide a trained model and test dataset to compute the confusion matrix.")

with tab4:
    st.header("System / Model Status")
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        metrics = json.load(open(metrics_path))
        st.json(metrics)
    else:
        st.info("No metrics.json found (model may not have been trained).")
    if st.button("Download current model"):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                st.download_button("Download model", f, file_name="model_latest.h5")
        else:
            st.error("No model file available.")
