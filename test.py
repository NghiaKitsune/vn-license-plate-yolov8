"""
Test nhanh pipeline nhan dien bien so xe.
Chay: python test.py [--source anh.jpg | video.mp4 | 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from plate_pipeline import (
    PlateOCR,
    PlatePreprocessor,
    format_plate,
    is_valid,
    run_on_frame,
)


def resolve_default_model() -> str:
    root = Path("runs")
    candidates = [
        path for path in root.rglob("best.pt")
        if path.parent.name == "weights" and path.parent.parent.name.startswith("bien_so_v")
    ]
    if not candidates:
        return "runs/detect/runs/bien_so_v1/weights/best.pt"
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


MODEL_PATH = resolve_default_model()


def test_image(path: str, detector, prep: PlatePreprocessor, ocr: PlateOCR) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[LOI] Khong doc duoc: {path}")
        sys.exit(1)
    detections = run_on_frame(frame, detector, prep, ocr, conf_threshold=0.4)
    if not detections:
        print("  Khong phat hien bien so nao.")
        return
    for box, text, det_conf, ocr_conf in detections:
        fp = format_plate(text)
        status = "VALID" if is_valid(text) else "invalid"
        print(f"  [{status}] {fp}  (det {det_conf:.1%} | ocr {ocr_conf:.1%})")
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, fp, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    out = Path(path).with_stem(Path(path).stem + "_test")
    cv2.imwrite(str(out), frame)
    print(f"[OK] Luu ket qua: {out}")


def test_webcam_single_frame(detector, prep: PlatePreprocessor, ocr: PlateOCR) -> None:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[LOI] Khong mo duoc webcam.")
        sys.exit(1)
    detections = run_on_frame(frame, detector, prep, ocr, conf_threshold=0.4)
    for box, text, det_conf, ocr_conf in detections:
        fp = format_plate(text)
        print(f"  Phat hien: {fp}  (det {det_conf:.1%} | ocr {ocr_conf:.1%})")
    cv2.imwrite("ket_qua.jpg", frame)
    print("[OK] Luu: ket_qua.jpg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam",
                        help="Duong dan anh, hoac 'webcam'")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    print(f"[1/3] Load model: {args.model}")
    detector = YOLO(args.model)

    print("[2/3] Khoi tao OCR...")
    prep = PlatePreprocessor()
    ocr  = PlateOCR()

    print("[3/3] Chay test...\n")
    if args.source == "webcam":
        test_webcam_single_frame(detector, prep, ocr)
    else:
        test_image(args.source, detector, prep, ocr)


if __name__ == "__main__":
    main()
