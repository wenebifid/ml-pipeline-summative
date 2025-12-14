# src/model.py
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.class_names import CLASS_NAMES


def build_transfer_model(num_classes, input_shape=(224, 224, 3)):
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(
    train_gen,
    val_gen,
    out_path="models/model_latest.h5",
    epochs=10,
    class_names_path="models/class_names.json"
):
    # 🔒 Enforce class count from CLASS_NAMES (not generator guesswork)
    assert train_gen.num_classes == len(CLASS_NAMES), (
        "Mismatch between train generator classes and CLASS_NAMES"
    )

    model = build_transfer_model(len(CLASS_NAMES))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        ),
        ModelCheckpoint(
            out_path,
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # 💾 Save model
    model.save(out_path)

    # 💾 Save class names (CRITICAL)
    with open(class_names_path, "w") as f:
        json.dump(CLASS_NAMES, f)

    return model, history
