"""
Demo nhan dien bien so xe Viet Nam - real-time / video / anh tinh.

Cach dung:
  python demo.py                        # webcam mac dinh
  python demo.py --source video.mp4     # file video
  python demo.py --source anh.jpg       # anh tinh (xu ly 1 frame)
  python demo.py --source 1             # webcam thu 2

Phim tat khi xem video/webcam:
  Q / ESC  -- thoat
  S        -- chup frame hien tai
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from plate_pipeline import (
    PlateOCR,
    PlatePreprocessor,
    PlateTracker,
    format_plate,
    is_valid,
    run_on_frame,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_FILE   = "log_bien_so.csv"
CONF_THRESH = 0.45          # nguong detection
VOTES      = 3              # so vote toi thieu de chot bien so
FONT       = cv2.FONT_HERSHEY_SIMPLEX


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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_box(frame, box, label: str, confirmed: bool) -> None:
    x1, y1, x2, y2 = box
    color = (0, 220, 0) if confirmed else (0, 165, 255)   # xanh / cam
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), FONT, 0.65, (0, 0, 0), 2)


def _overlay_status(frame, n_detected: int, n_logged: int) -> None:
    h = frame.shape[0]
    cv2.putText(frame, f"Dang detect: {n_detected} | Da log: {n_logged}",
                (10, h - 15), FONT, 0.55, (255, 255, 0), 1)
    cv2.putText(frame, "Q/ESC: thoat | S: chup",
                (10, h - 35), FONT, 0.45, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_image(source: str, detector, preprocessor, ocr: PlateOCR) -> None:
    frame = cv2.imread(source)
    if frame is None:
        print(f"[LOI] Khong doc duoc anh: {source}")
        sys.exit(1)

    detections = run_on_frame(frame, detector, preprocessor, ocr, CONF_THRESH)
    for (box, text, det_conf, ocr_conf) in detections:
        label = (
            f"{format_plate(text)} ocr:{ocr_conf:.0%}"
            if text else f"det:{det_conf:.0%}"
        )
        _draw_box(frame, box, label, confirmed=True)
        print(
            f"  Bien so: {format_plate(text)}  |  "
            f"Det: {det_conf:.1%} | OCR: {ocr_conf:.1%}"
        )

    out_path = Path(source).with_stem(Path(source).stem + "_result")
    cv2.imwrite(str(out_path), frame)
    print(f"[OK] Da luu: {out_path}")
    cv2.imshow("Ket qua", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Video / webcam mode
# ---------------------------------------------------------------------------

def run_video(source, detector, preprocessor, ocr: PlateOCR) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LOI] Khong mo duoc nguon: {source}")
        sys.exit(1)

    tracker = PlateTracker(votes_required=VOTES)
    logged_plates: set[str] = set()

    log_f = open(LOG_FILE, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["Thoi gian", "Bien so", "Do chinh xac"])

    print(f"[INFO] Bat dau xu ly: {source}")
    print("[INFO] Nhan Q hoac ESC de thoat, S de chup anh")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        raw_detections = run_on_frame(frame, detector, preprocessor, ocr, CONF_THRESH)
        tracking_input = [
            (
                box,
                text,
                max(0.2, det_conf) + (ocr_conf * 1.4) + (1.0 if is_valid(text) else 0.0),
            )
            for box, text, det_conf, ocr_conf in raw_detections
        ]
        conf_map = {box: (det_conf, ocr_conf) for box, _, det_conf, ocr_conf in raw_detections}

        tracked = tracker.update(tracking_input)

        for tid, box, text, confirmed in tracked:
            det_conf, ocr_conf = conf_map.get(box, (0.0, 0.0))
            display = format_plate(text) if text else ""
            label = f"{display} ocr:{ocr_conf:.0%}" if display else f"det:{det_conf:.0%}"
            _draw_box(frame, box, label, confirmed)

            if confirmed and not tracker.is_logged(tid):
                formatted = format_plate(text)
                tracker.mark_logged(tid)
                if formatted not in logged_plates:
                    logged_plates.add(formatted)
                    ts = datetime.now().strftime("%H:%M:%S")
                    writer.writerow([ts, formatted, f"{ocr_conf:.0%}"])
                    log_f.flush()
                    print(f"[{ts}] LOG: {formatted}  (det {det_conf:.0%} | ocr {ocr_conf:.0%})")

        _overlay_status(frame, len(tracked), len(logged_plates))
        cv2.imshow("Nhan Dien Bien So Xe", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q hoac ESC
            break
        if key == ord("s"):
            snap = f"snapshot_{frame_id}.jpg"
            cv2.imwrite(snap, frame)
            print(f"[SNAP] Da luu: {snap}")

    cap.release()
    log_f.close()
    cv2.destroyAllWindows()
    print(f"\n[XONG] Da log {len(logged_plates)} bien so -> {LOG_FILE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Nhan dien bien so xe Viet Nam")
    parser.add_argument(
        "--source", default="0",
        help="Nguon video: 0=webcam, duong dan video, duong dan anh (default: 0)"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Duong dan model YOLO (default: {MODEL_PATH})"
    )
    args = parser.parse_args()

    print("[1/3] Dang load model YOLO...")
    detector = YOLO(args.model)

    print("[2/3] Dang khoi tao OCR (PaddleOCR)...")
    preprocessor = PlatePreprocessor()
    ocr = PlateOCR()

    print("[3/3] Bat dau nhan dien...\n")

    source = args.source
    # Kiem tra xem source co phai anh tinh khong
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if Path(source).suffix.lower() in img_exts:
        run_image(source, detector, preprocessor, ocr)
    else:
        # Video hoac webcam
        try:
            source = int(source)
        except ValueError:
            pass
        run_video(source, detector, preprocessor, ocr)


if __name__ == "__main__":
    main()
