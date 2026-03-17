"""
Drowsiness Detection System - CNN Model Training
================================================
This script trains a CNN model to classify eyes as Open or Closed.
Dataset Structure Required:
    dataset/
        train/
            Open/   (eye images when open)
            Closed/ (eye images when closed)
        test/
            Open/
            Closed/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE    = (24, 24)
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
DATASET_DIR = "dataset"
MODEL_PATH  = "model/drowsiness_model.h5"


# ── Data Augmentation & Generators ───────────────────────────────────────────
def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, test_gen


# ── CNN Architecture ──────────────────────────────────────────────────────────
def build_model():
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(*IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')   # 0=Closed, 1=Open
    ])
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    os.makedirs("model", exist_ok=True)

    print("[INFO] Loading data generators …")
    train_gen, test_gen = get_data_generators()
    print(f"       Classes: {train_gen.class_indices}")

    print("[INFO] Building model …")
    model = build_model()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True,
                        monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=7, monitor='val_accuracy',
                      restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3,
                          monitor='val_loss', verbose=1)
    ]

    print("[INFO] Training …")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating …")
    test_gen.reset()
    preds = (model.predict(test_gen) > 0.5).astype(int).flatten()
    labels = test_gen.classes
    target_names = list(test_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))

    # ── Plot Training Curves ───────────────────────────────────────────────────
    _plot_history(history)
    print(f"\n[INFO] Model saved to: {MODEL_PATH}")


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],    label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'],    label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model/training_curves.png")
    plt.show()
    print("[INFO] Training curves saved to model/training_curves.png")


if __name__ == "__main__":
    train()
