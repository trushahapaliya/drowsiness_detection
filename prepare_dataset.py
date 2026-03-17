"""
Dataset Preparation Script
===========================
Organises raw eye images into the expected folder structure and optionally
augments the dataset.

Expected input structure (any layout):
    raw_data/
        open/   *.jpg / *.png
        closed/ *.jpg / *.png

Output structure:
    dataset/
        train/
            Open/
            Closed/
        test/
            Open/
            Closed/

Usage:
    python prepare_dataset.py --src raw_data --split 0.8
"""

import os
import shutil
import argparse
import random
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   default="raw_data",
                    help="Source directory with open/ and closed/ sub-folders")
    ap.add_argument("--dst",   default="dataset",
                    help="Destination dataset directory")
    ap.add_argument("--split", type=float, default=0.8,
                    help="Train split ratio (default 0.8)")
    ap.add_argument("--seed",  type=int,   default=42)
    return ap.parse_args()


def prepare(src, dst, split, seed):
    random.seed(seed)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls_name in ["open", "closed"]:
        src_dir = Path(src) / cls_name
        if not src_dir.exists():
            print(f"[WARN] Source folder not found: {src_dir}")
            continue

        files = [f for f in src_dir.iterdir() if f.suffix.lower() in exts]
        random.shuffle(files)

        n_train = int(len(files) * split)
        splits  = {"train": files[:n_train], "test": files[n_train:]}
        cap_cls = cls_name.capitalize()

        for split_name, file_list in splits.items():
            out_dir = Path(dst) / split_name / cap_cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in file_list:
                shutil.copy2(f, out_dir / f.name)

        print(f"[OK] {cap_cls}: {n_train} train, "
              f"{len(files)-n_train} test images")

    print(f"\n[INFO] Dataset ready at: {dst}/")


if __name__ == "__main__":
    args = prepare_dataset = parse_args()
    prepare(args.src, args.dst, args.split, args.seed)
