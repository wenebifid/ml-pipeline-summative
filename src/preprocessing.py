import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.class_names import CLASS_NAMES  # 👈 ADD THIS

IMG_SIZE = (224, 224)

def create_generators(train_dir, val_dir, batch_size=32, seed=42):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,   # 👈 ENFORCE ORDER
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,   # 👈 SAME ORDER
        shuffle=False
    )

    return train_gen, val_gen


def load_image_for_prediction(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr
