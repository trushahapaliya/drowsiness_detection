"""
Model Evaluation Script
========================
Evaluates a trained model on the test set and shows a confusion matrix,
sample predictions, and per-class metrics.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model model/drowsiness_model.h5
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="model/drowsiness_model.h5")
    ap.add_argument("--dataset", default="dataset/test")
    return ap.parse_args()


def evaluate(model_path, dataset_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    print("[INFO] Loading model …")
    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(24, 24),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    print("[INFO] Running predictions …")
    probs  = model.predict(gen, verbose=1).flatten()
    preds  = (probs > 0.5).astype(int)
    labels = gen.classes
    names  = list(gen.class_indices.keys())

    # ── Classification Report ─────────────────────────────────────────────────
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=names))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='darkorange')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    out = "model/evaluation.png"
    plt.savefig(out)
    plt.show()
    print(f"\n[INFO] Evaluation charts saved to: {out}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.dataset)
