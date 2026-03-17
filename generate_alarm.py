"""
Alarm Sound Generator
=====================
Generates a simple beep alarm.wav using only Python standard library (wave + math).
Run once to create the sounds/alarm.wav file.

Usage:
    python generate_alarm.py
"""

import wave
import struct
import math
import os

SAMPLE_RATE = 44100
DURATION    = 1.0      # seconds per beep cycle
FREQUENCY   = 880      # Hz  (A5 note)
AMPLITUDE   = 32000
OUTPUT_PATH = "sounds/alarm.wav"


def generate_alarm(path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_samples = int(SAMPLE_RATE * DURATION)
    # Two-tone beep: 880 Hz on, then silence
    samples = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # First half: tone; second half: silence
        if t < DURATION / 2:
            v = int(AMPLITUDE * math.sin(2 * math.pi * FREQUENCY * t))
        else:
            v = 0
        samples.append(v)

    # Repeat 3 times
    samples = samples * 3

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    print(f"[OK] Alarm sound saved to: {path}")


if __name__ == "__main__":
    generate_alarm()
