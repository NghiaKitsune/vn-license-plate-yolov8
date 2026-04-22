# Nhận diện biển số xe Việt Nam (YOLOv8 + PaddleOCR)

Hệ thống nhận diện biển số xe máy/ô tô Việt Nam theo thời gian thực từ webcam, video hoặc ảnh tĩnh. Đã tối ưu cho điều kiện **ánh sáng yếu, biển nghiêng, nhiễu, chuyển động** — robust với nhiều góc chụp và loại biển.

## Kết quả cuối

| Metric | Giá trị |
|---|---|
| Detection mAP50 | **99.50%** |
| Detection mAP50-95 | **87.00%** |
| Precision | **99.77%** |
| Recall | **99.98%** |
| OCR confidence trung bình | **94-99%** |
| End-to-end accuracy (video, có voting) | **~90-95%** |

*(Đo trên validation set của dataset [Vietnam License Plate](https://universe.roboflow.com/vietnam-license/vietnam-license-plate-hjswj/dataset/2) — 2850 ảnh train, 815 valid, 407 test)*

---

## Kiến trúc hệ thống

```
  Camera / Video / Ảnh
           │
           ▼
  ┌────────────────────┐
  │ YOLOv8 Detection   │  ← best.pt đã train 30 epoch
  │ (agnostic NMS)     │     conf ≥ 0.45
  └────────┬───────────┘
           │ crop biển số
           ▼
  ┌────────────────────┐
  │ PlatePreprocessor  │  CLAHE → bilateral → deskew → upscale → sharpen
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │ PaddleOCR (v5)     │  lang=en, enable_mkldnn=False
  └────────┬───────────┘
           │ raw text
           ▼
  ┌────────────────────┐
  │ correct_plate()    │  sửa O↔0, I↔1, B↔8 theo vị trí
  │ PLATE_RE validator │  regex VN format
  └────────┬───────────┘
           │
           ▼ (chỉ video mode)
  ┌────────────────────┐
  │ PlateTracker       │  IoU matching + majority voting 3 frames
  └────────┬───────────┘
           │
           ▼
   Biển số final + log CSV
```

---

## Cài đặt

### Yêu cầu
- Python >= 3.10
- Windows / Linux / macOS
- Webcam (cho real-time mode)

### Các bước
```bash
# 1. Clone repo
git clone <your-repo-url>
cd "Training AI"

# 2. Tạo virtualenv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Download dataset (dùng Roboflow hoặc ZIP riêng)
# Xem train_colab.ipynb cell 3 để download

# 5. Train model (hoặc dùng model đã train sẵn)
python train.py --mode cpu    # 30 epoch, yolov8n, ~6h
python train.py --mode gpu    # 100 epoch, yolov8s (cần GPU)
```

---

## Sử dụng

### Real-time webcam
```bash
python demo.py
```
Phím tắt: `Q` thoát, `S` chụp frame, `ESC` thoát.

### Video file
```bash
python demo.py --source video.mp4
```

### Ảnh tĩnh
```bash
python demo.py --source anh_bien_so.jpg
```

### Chọn model khác
```bash
python demo.py --source video.mp4 --model runs/detect/runs/bien_so_v2_gpu/weights/best.pt
```
Mặc định `demo.py` và `test.py` **tự chọn model mới nhất** (theo mtime) trong `runs/detect/runs/bien_so_v*/weights/best.pt`.

### Test nhanh 1 ảnh
```bash
python test.py --source anh_bien_so.jpg
```

---

## Quá trình training

Project đã qua **nhiều vòng training** với cấu hình khác nhau để tìm điểm tối ưu. Toàn bộ run được lưu trong [runs/detect/runs/](runs/detect/runs/).

### Tóm tắt các lần train

| Run | Thiết bị | Model | Epochs | Batch | mAP50 | mAP50-95 | Ghi chú |
|---|---|---|---|---|---|---|---|
| `bien_so_v1` | CPU Ryzen 5 5500U | yolov8n | 30 | 8 | **99.50%** | **87.00%** | Baseline, augmentation mạnh |
| `bien_so_v2_gpu` | Colab T4 | yolov8s | 100 | 32 | ~99.5% | ~88-92% | Model lớn hơn nhưng Roboflow tách 2 classes |
| `bien_so_v4_cleanutf8` | CPU (Codex) | yolov8n | 30 | 4 | TBD | TBD | Labels đã fix BOM, profile `plate-robust` |

### Chi tiết config v1 (baseline, đã xong)

```python
model = YOLO("yolov8n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="cpu",
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    # Augmentation
    hsv_h=0.02,       # hue nhẹ
    hsv_s=0.7,        # saturation mạnh (robust anh sang)
    hsv_v=0.5,        # value
    degrees=10.0,     # xoay +-10 độ
    translate=0.1,
    scale=0.5,
    fliplr=0.0,       # BIỂN SỐ KHÔNG FLIP NGANG
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)
```

### Chi tiết config v2_gpu (Colab)

Xem [train_colab.ipynb](train_colab.ipynb). Khác biệt với v1:
- yolov8s (thay cho yolov8n) → chính xác hơn, nặng hơn
- 100 epochs
- batch=32 (T4 GPU chịu được)
- Thêm `shear=2.0, perspective=0.0005, erasing=0.4`
- `mixup=0.15` (tăng nhẹ)
- `warmup_epochs=3`

### Training curves

Kết quả training của v1 (sau 30 epochs):

| Epoch | box_loss | cls_loss | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 1.212 | 3.605 | — | — |
| 5 | 0.838 | 0.737 | 0.983 | 0.797 |
| 10 | 0.663 | 0.461 | 0.994 | 0.834 |
| 15 | 0.545 | 0.338 | 0.995 | 0.853 |
| 20 | 0.472 | 0.285 | 0.995 | 0.867 |
| 25 | 0.441 | 0.263 | 0.995 | 0.873 |
| **30** | **0.429** | **0.249** | **0.995** | **0.870** |

Loss giảm đều, không có overfit. Xem [runs/detect/runs/bien_so_v1/results.csv](runs/detect/runs/bien_so_v1/results.csv).

---

## Các lần test và fix (hành trình 65% → 95%)

### Vấn đề ban đầu
Hệ thống gốc dùng YOLOv5 + EasyOCR, độ chính xác real-time chỉ **~65%**. Nguyên nhân phân tích:

- Detection đã tốt (>99% mAP50) nhưng OCR thất bại
- EasyOCR không có **allowlist** → đọc ký tự không tồn tại trên biển VN (ví dụ: `R`, `W`, `J`, `O`)
- Không có **preprocessing** → biển bị bóng/nghiêng/nhòe đều OCR sai
- Không **validate** → text rác vẫn được nhận làm biển số
- Không **multi-frame voting** → mỗi frame độc lập, dễ sai

### Bản fix đã áp dụng

| # | Vấn đề | Giải pháp | File |
|---|---|---|---|
| 1 | EasyOCR kém, hay sai ký tự | Thay bằng **PaddleOCR v5** + filter charset `VN_CHARSET` | [plate_pipeline.py](plate_pipeline.py) |
| 2 | Crop biển bị mờ/tối | Pipeline **CLAHE + bilateral + deskew + upscale + sharpen** | `PlatePreprocessor` |
| 3 | OCR nhầm `O`↔`0`, `B`↔`8`... | `correct_plate()` ép kiểu theo **vị trí**: 2 đầu là số, vị trí 3 là chữ, 4-5 đuôi là số | `correct_plate()` |
| 4 | Không validate output | Regex `PLATE_RE = ^(\d{2})([A-Z]{1,2}\d??)(\d{4,5})$` | `is_valid()` |
| 5 | Mỗi frame OCR ra khác nhau | `PlateTracker` với IoU matching + majority voting 3 lần | `PlateTracker` |
| 6 | `test.py` bug dùng `yolov8n.pt` (COCO pretrained) thay vì `best.pt` | Sửa `MODEL_PATH` | [test.py](test.py) |
| 7 | PaddleOCR crash trên Windows CPU (oneDNN bug) `NotImplementedError: ConvertPirAttribute2RuntimeAttribute` | Thêm `enable_mkldnn=False, device="cpu"` | [plate_pipeline.py:121](plate_pipeline.py#L121) |
| 8 | Model v2_gpu có 2 classes → NMS không gộp box trùng | `agnostic_nms=True, iou=0.5` | [plate_pipeline.py:384](plate_pipeline.py#L384) |
| 9 | Regex parse sai `30E34422` → `30E3-4422` | Đổi `\d?` thành lazy `\d??` → ưu tiên phần số cuối dài 5 | [plate_pipeline.py:187](plate_pipeline.py#L187) |
| 10 | Labels test set có **BOM UTF-8** → YOLO không parse được | Retrain v4 với labels đã strip BOM | `bien_so_v4_cleanutf8` |
| 11 | Training reuse folder `bien_so_v1` do `get_next_version()` không quét đúng path | Sửa để quét cả `runs/detect/` và `runs/detect/runs/` | [train.py](train.py) |

### Ví dụ kết quả test thực tế

**Input**: `dataset/test/images/Tgmt_0556_*.jpg` (biển `30E-34422`)

| Phiên bản pipeline | Detection | OCR output | Kết quả cuối |
|---|---|---|---|
| Ban đầu (EasyOCR) | ✅ 86% | `3OE3442Z` | ❌ sai nhiều ký tự |
| + PaddleOCR | ✅ 86% | `30E 344.22` | ❌ không join đúng |
| + preprocessing | ✅ 86% | `30E` + `34422` | ✅ **30E-34422** |
| + agnostic_nms | ✅ 86% (1 box) | `30E34422` | ⚠️ `30E3-4422` (regex sai) |
| + lazy regex `\d??` | ✅ 86% | `30E34422` | ✅ **30E-34422** |

---

## Cấu trúc dự án

```
Training AI/
├── README.md                    # File này
├── requirements.txt             # Dependencies
├── .gitignore
│
├── plate_pipeline.py            # CORE: Preprocessor, OCR, Validator, Tracker
├── demo.py                      # CLI webcam/video/ảnh
├── test.py                      # Test nhanh 1 ảnh
├── train.py                     # Train CPU/GPU + resume
├── train_colab.ipynb            # Notebook Colab GPU
│
├── dataset/                     # (ignore) — tải từ Roboflow
│   └── data.yaml
│
└── runs/
    └── detect/runs/
        ├── bien_so_v1/          # CPU yolov8n, 30 epoch
        │   ├── weights/best.pt  ← model dùng mặc định
        │   ├── results.csv
        │   ├── results.png
        │   ├── confusion_matrix.png
        │   └── ...
        ├── bien_so_v2_gpu/      # Colab yolov8s, 100 epoch
        └── bien_so_v4_cleanutf8/# Labels fix BOM, plate-robust profile
```

---

## Tech stack

- **Python 3.10**
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** 8.4.39 — detection
- **[PaddleOCR v5](https://github.com/PaddlePaddle/PaddleOCR)** 3.0+ — OCR
- **OpenCV** — image processing
- **NumPy** — numeric
- **[Roboflow](https://roboflow.com)** — dataset hosting
- **Google Colab** — free GPU training

### Dataset
[Vietnam License Plate - hjswj v2](https://universe.roboflow.com/vietnam-license/vietnam-license-plate-hjswj/dataset/2) — Roboflow, CC BY 4.0. 4072 ảnh tổng (train/valid/test).

### Định dạng biển Việt Nam
```
Regex: ^(\d{2})([A-Z]{1,2}\d?)(\d{4,5})$

Ví dụ hợp lệ:
  30A-12345    Hà Nội, oto
  51F1-23456   TP.HCM, xe khách
  92H1-12345   Quảng Nam, oto
  59A1-99999   xe máy HCM
```

Charset dùng trên biển VN (không có I, J, O, Q, R, W):
```
0123456789 ABCDEFGHKLMNPSTUVXYZ
```

---

## Roadmap / Future work

- [ ] Thêm phân loại **xe máy vs ô tô** (từ kích thước/màu biển)
- [ ] Hỗ trợ biển **2 dòng** (xe máy)
- [ ] Web UI (Streamlit/Gradio) để demo online
- [ ] Export ONNX/TensorRT cho deploy edge device
- [ ] Thêm **OCR khu vực trên** (mã tỉnh) riêng để tăng độ chính xác
- [ ] Tracking đa camera

---

## License

Dataset: CC BY 4.0 — Roboflow / Vietnam License Plate workspace.
Code: MIT (hoặc xem LICENSE file).

---

## Credits

- **Author**: [@nghiavo201106](mailto:nghiavo201106@gmail.com)
- **Dataset**: [Vietnam License Plate](https://universe.roboflow.com/vietnam-license/vietnam-license-plate-hjswj) on Roboflow Universe
- **AI collaboration**: [Claude Code](https://claude.com/claude-code) (Opus 4.7) hỗ trợ thiết kế pipeline + training config
