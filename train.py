"""
Train YOLOv8 nhan dien bien so xe Viet Nam.

CPU  (may nay): python train.py --mode cpu      # ~6h, 30 epochs, yolov8n
GPU  (Colab)  : python train.py --mode gpu      # ~1-2h, 100 epochs, yolov8s
Resume        : python train.py --resume        # tiep tuc tu checkpoint
Smoke test    : python train.py --epochs 3 --name bien_so_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DATA = "dataset/data.yaml"
PROJECT = Path("runs")
BASE_NAME = "bien_so"


def _iter_run_dirs() -> list[Path]:
    if not PROJECT.exists():
        return []
    return [
        d for d in PROJECT.rglob(f"{BASE_NAME}_v*")
        if d.is_dir()
    ]


def get_next_version() -> str:
    existing = [d.name for d in _iter_run_dirs()]
    versions = []
    for n in existing:
        try:
            versions.append(int(n.split("_v")[-1]))
        except ValueError:
            pass
    next_v = max(versions, default=0) + 1
    return f"{BASE_NAME}_v{next_v}"


def _defaults_for_mode(mode: str) -> dict[str, int | float | str]:
    if mode == "cpu":
        return {
            "model": "yolov8n.pt",
            "epochs": 30,
            "batch": 8,
            "patience": 15,
            "save_period": 5,
        }
    return {
        "model": "yolov8s.pt",
        "epochs": 100,
        "batch": 16,
        "patience": 20,
        "save_period": 10,
    }


def _resolve_training_args(args: argparse.Namespace) -> None:
    defaults = _defaults_for_mode(args.mode)
    if args.model is None:
        args.model = defaults["model"]
    if args.epochs is None:
        args.epochs = int(defaults["epochs"])
    if args.batch is None:
        args.batch = int(defaults["batch"])
    if args.patience is None:
        args.patience = int(defaults["patience"])
    if args.save_period is None:
        args.save_period = int(defaults["save_period"])


def _augmentation_profile(mode: str, profile: str) -> dict[str, float]:
    if profile == "baseline":
        if mode == "cpu":
            return {
                "hsv_h": 0.02,
                "hsv_s": 0.7,
                "hsv_v": 0.5,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.5,
                "fliplr": 0.0,
                "mosaic": 1.0,
                "mixup": 0.1,
                "copy_paste": 0.1,
            }
        return {
            "hsv_h": 0.02,
            "hsv_s": 0.7,
            "hsv_v": 0.5,
            "degrees": 15.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 2.0,
            "perspective": 0.0005,
            "fliplr": 0.0,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.1,
            "erasing": 0.4,
        }

    # Profile tinh chinh cho bien so: giu hinh hoc that hon, giam phep ghep anh
    if mode == "cpu":
        return {
            "hsv_h": 0.015,
            "hsv_s": 0.5,
            "hsv_v": 0.35,
            "degrees": 7.0,
            "translate": 0.06,
            "scale": 0.35,
            "fliplr": 0.0,
            "mosaic": 0.5,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.15,
        }
    return {
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.35,
        "degrees": 10.0,
        "translate": 0.08,
        "scale": 0.4,
        "shear": 1.0,
        "perspective": 0.0003,
        "fliplr": 0.0,
        "mosaic": 0.6,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.2,
    }


def _build_train_kwargs(
    args: argparse.Namespace,
    name: str,
    device: str | int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "data": DATA,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "patience": args.patience,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "cos_lr": True,
        "warmup_epochs": 3,
        "workers": args.workers,
        "cache": args.cache,
        "close_mosaic": args.close_mosaic,
        "save": True,
        "save_period": args.save_period,
        "project": str(PROJECT),
        "name": name,
        "exist_ok": True,
        "plots": True,
    }
    kwargs.update(_augmentation_profile(args.mode, args.profile))
    return kwargs


def train_cpu(name: str, args: argparse.Namespace) -> None:
    """Train tren CPU voi profile co the tinh chinh tu command line."""
    model = YOLO(args.model)
    model.train(**_build_train_kwargs(args, name, "cpu"))


def train_gpu(name: str, args: argparse.Namespace) -> None:
    """Train tren GPU (Colab/local GPU) voi profile tinh chinh."""
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    model.train(**_build_train_kwargs(args, name, device))


def find_latest_checkpoint(weight_name: str = "last.pt") -> Path | None:
    checkpoints: list[Path] = []
    for run_dir in _iter_run_dirs():
        candidate = run_dir / "weights" / weight_name
        if candidate.exists():
            checkpoints.append(candidate)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def resume_training() -> None:
    """Tiep tuc tu last.pt cua run moi nhat."""
    last_pt = find_latest_checkpoint("last.pt")
    if last_pt is None:
        print(f"[LOI] Khong tim thay last.pt de resume trong {PROJECT}")
        return
    print(f"[INFO] Resume tu: {last_pt}")
    model = YOLO(str(last_pt))
    model.train(resume=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["cpu", "gpu"], default="cpu",
        help="cpu: 30 epochs yolov8n | gpu: 100 epochs yolov8s (default: cpu)"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--save-period", type=int, default=None)
    parser.add_argument("--close-mosaic", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--profile",
        choices=["baseline", "plate-robust"],
        default="plate-robust",
        help="baseline: giong cau hinh cu | plate-robust: giam augment kho do cho bien so"
    )
    parser.add_argument("--model", default=None, help="Model goc, vd yolov8n.pt")
    parser.add_argument("--name", default=None, help="Ten run. Mac dinh tu dong tang version")
    parser.add_argument(
        "--cache", action="store_true",
        help="Bat cache dataset de train CPU on dinh hon"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Tiep tuc tu checkpoint cuoi cung"
    )
    args = parser.parse_args()

    if args.resume:
        resume_training()
        return

    _resolve_training_args(args)
    name = args.name or get_next_version()
    print(f"[INFO] Ten run: {name}")
    print(f"[INFO] Mode: {args.mode.upper()}")
    print(f"[INFO] Model goc: {args.model}")
    print(f"[INFO] Profile: {args.profile}")
    print(f"[INFO] Save dir: {PROJECT / name}")

    if args.mode == "cpu":
        print(f"[INFO] CPU mode: {args.epochs} epochs, batch {args.batch}")
        print("[TIP] De trai nghiem tot hon, chay tren Google Colab GPU:")
        print("      python train.py --mode gpu")
        train_cpu(name, args)
    else:
        train_gpu(name, args)

    print("\n" + "=" * 55)
    print(f"TRAIN XONG! Model: {PROJECT / name / 'weights' / 'best.pt'}")
    print("=" * 55)
    print("\n[TIP] De dung model moi trong demo:")
    print(f"  python demo.py --model {PROJECT / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
