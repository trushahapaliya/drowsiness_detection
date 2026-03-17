"""
Drowsiness Detection System - Real-Time Detection
==================================================
Uses webcam + Haar Cascades + trained CNN to detect driver drowsiness.
Triggers an alarm when eyes are closed for a sustained period.

Usage:
    python detect_drowsiness.py
    python detect_drowsiness.py --camera 0
    python detect_drowsiness.py --threshold 15
"""

import cv2
import numpy as np
import argparse
import time
import os
from tensorflow.keras.models import load_model
from utils.alarm import AlarmPlayer
from utils.eye_aspect_ratio import get_eye_region
import warnings
warnings.filterwarnings('ignore')


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Real-Time Drowsiness Detection")
    ap.add_argument("--camera",    type=int,   default=0,
                    help="Camera index (default 0)")
    ap.add_argument("--threshold", type=int,   default=15,
                    help="Consecutive closed-eye frames before alarm (default 15)")
    ap.add_argument("--model",     type=str,   default="model/drowsiness_model.h5",
                    help="Path to trained Keras model")
    ap.add_argument("--alarm",     type=str,   default="sounds/alarm.wav",
                    help="Path to alarm sound file")
    ap.add_argument("--no-alarm",  action="store_true",
                    help="Disable alarm sound")
    return ap.parse_args()


# ── Haar Cascade Loader ───────────────────────────────────────────────────────
def load_cascades():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascades. "
                           "Check your OpenCV installation.")
    return face_cascade, eye_cascade


# ── Preprocess Eye Region for CNN ────────────────────────────────────────────
def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, (24, 24))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) if len(eye.shape) == 3 else eye
    eye = eye / 255.0
    eye = eye.reshape(1, 24, 24, 1)
    return eye


# ── Overlay Helpers ───────────────────────────────────────────────────────────
def draw_status(frame, status, score, threshold, fps):
    h, w = frame.shape[:2]

    # Status bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Status text
    color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Score bar
    bar_w = int((score / threshold) * 200)
    bar_w = min(bar_w, 200)
    cv2.rectangle(frame, (10, 35), (210, 50), (50, 50, 50), -1)
    bar_color = (0, 255, 0) if score < threshold * 0.6 else \
                (0, 165, 255) if score < threshold else (0, 0, 255)
    cv2.rectangle(frame, (10, 35), (10 + bar_w, 50), bar_color, -1)
    cv2.putText(frame, f"Score:{score}/{threshold}", (220, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def draw_alarm_banner(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "⚠  DROWSINESS ALERT! WAKE UP! ⚠",
                (w // 2 - 220, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ── Main Detection Loop ───────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found at '{args.model}'.")
        print("        Run  train_model.py  first, or use  generate_demo_model.py")
        print("        to create a demo model without a dataset.")
        return

    print("[INFO] Loading CNN model …")
    model = load_model(args.model)

    # Load cascades
    print("[INFO] Loading Haar Cascades …")
    face_cascade, eye_cascade = load_cascades()

    # Alarm
    alarm = AlarmPlayer(args.alarm) if not args.no_alarm else None

    # Camera
    print(f"[INFO] Opening camera {args.camera} …")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    score      = 0        # consecutive closed-eye frames
    alarm_on   = False
    prev_time  = time.time()

    print("[INFO] Detection running. Press 'q' to quit, 'r' to reset score.")
    print(f"       Alarm threshold: {args.threshold} frames\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying …")
            continue

        # FPS
        cur_time = time.time()
        fps      = 1.0 / max(cur_time - prev_time, 1e-6)
        prev_time = cur_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        status      = "NO FACE"
        eyes_closed = False

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                          (255, 200, 0), 2)

            roi_gray  = gray[fy:fy + fh, fx:fx + fw]
            roi_color = frame[fy:fy + fh, fx:fx + fw]

            # Detect eyes within face ROI
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(20, 20), maxSize=(80, 80)
            )

            eye_preds = []
            for (ex, ey, ew, eh) in eyes[:2]:   # process at most 2 eyes
                eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                if eye_img.size == 0:
                    continue

                processed = preprocess_eye(eye_img)
                pred      = model.predict(processed, verbose=0)[0][0]
                # pred ≈ 1 → Open,  pred ≈ 0 → Closed
                eye_preds.append(pred)

                # Draw eye box
                eye_color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
                label     = f"Open {pred:.2f}" if pred > 0.5 else f"Closed {1-pred:.2f}"
                cv2.rectangle(roi_color,
                              (ex, ey), (ex + ew, ey + eh), eye_color, 2)
                cv2.putText(roi_color, label, (ex, ey - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_color, 1)

            if eye_preds:
                avg_pred     = np.mean(eye_preds)
                eyes_closed  = avg_pred < 0.5
                status       = "DROWSY" if eyes_closed else "AWAKE"
            else:
                status = "EYES N/A"

            break   # process first detected face only

        # Update score
        if eyes_closed:
            score += 1
        else:
            score = max(0, score - 1)

        # Alarm logic
        if score >= args.threshold:
            alarm_on = True
            if alarm:
                alarm.play()
            draw_alarm_banner(frame)
        else:
            alarm_on = False
            if alarm:
                alarm.stop()

        draw_status(frame, status, score, args.threshold, fps)

        cv2.imshow("Drowsiness Detection System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            score = 0
            print("[INFO] Score reset.")

    if alarm:
        alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
