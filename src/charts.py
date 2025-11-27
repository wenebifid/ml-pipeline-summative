# src/charts.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from src.preprocessing import IMG_SIZE
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def plot_class_distribution(train_dir, out_path="models/class_distribution.png"):
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])
    counts = [len([f for f in os.listdir(os.path.join(train_dir,c)) if os.path.isfile(os.path.join(train_dir,c,f))]) for c in classes]
    plt.figure(figsize=(10,6))
    plt.bar(classes, counts)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path

def _load_image(path):
    img = image.load_img(path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return arr

def plot_confusion_matrix_png(model_path, test_dir, out_path="models/confusion_matrix.png"):
    model = tf.keras.models.load_model(model_path)
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir,d))])
    y_true = []
    y_pred = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(test_dir, cls)
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                arr = _load_image(fpath)
                pred = model.predict(np.expand_dims(arr,0))
                y_pred.append(np.argmax(pred))
                y_true.append(idx)
            except Exception:
                continue
    if len(y_true) == 0:
        raise ValueError("No predictions made; check test directory and model.")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path
