"""
Demo Model Generator
====================
Generates a pre-structured (random-weight) Keras model so you can test
detect_drowsiness.py immediately WITHOUT a dataset.

For accurate predictions, train a real model with train_model.py after
downloading an eye dataset (see README.md for links).

Usage:
    python generate_demo_model.py
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)

MODEL_PATH = "model/drowsiness_model.h5"


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(24, 24, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    print("[INFO] Building demo model (random weights) …")
    model = build_model()
    model.summary()
    model.save(MODEL_PATH)
    print(f"\n[INFO] Demo model saved to: {MODEL_PATH}")
    print("[WARN] This model has RANDOM weights — predictions will be unreliable.")
    print("       Train a real model with: python train_model.py")
