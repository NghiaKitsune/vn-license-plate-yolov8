# Hệ thống nhận diện biển số xe Việt Nam

> Dự án cá nhân nghiên cứu và triển khai hệ thống nhận diện biển số xe theo thời gian thực - tối ưu cho điều kiện **ánh sáng yếu, biển nghiêng, nhiễu và chuyển động**. Tác giả: **Nghia Vo** (nghiavo201106@gmail.com).

---

## 1. Tóm tắt kết quả

| Chỉ số | Mô hình v1 (CPU) | Mô hình v2 (Colab GPU) |
|---|---|---|
| Backbone | YOLOv8n | YOLOv8s |
| Epochs | 30 | 100 |
| Thời gian train | ~6 giờ | ~1.5 giờ |
| Kích thước model | 6.2 MB | 22.5 MB |
| **mAP50** | **99.50%** | ~99.5% |
| **mAP50-95** | **87.00%** | 88-92% |
| Precision | 99.77% | 99%+ |
| Recall | 99.98% | 99%+ |
| Inference trên CPU | ~25 ms/ảnh | ~100 ms/ảnh |

**Tích hợp end-to-end** (detection + OCR + tracking):
- Độ chính xác đọc biển số trên ảnh tĩnh: **~90%**
- Độ chính xác trên video (nhờ multi-frame voting): **~95%**
- Cải thiện so với pipeline gốc (EasyOCR, không preprocessing): **từ 65% lên 95%**

---

## 2. Động lực và bối cảnh

Khi bắt đầu, tôi có sẵn một pipeline YOLOv5 + EasyOCR cho ra kết quả chỉ **~65%** - quá thấp để dùng thực tế. Tôi quyết định tự phân tích từng khâu để tìm đúng nút thắt, thay vì chỉ "train lại cho nhiều epoch hơn". Mục tiêu cá nhân của tôi trong dự án này là:

1. **Hiểu được kiến trúc end-to-end**: từ detection, preprocessing, OCR đến post-processing và tracking.
2. **So sánh rõ hai hướng train**: chạy CPU local để kiểm chứng pipeline, chạy Colab GPU để đạt kết quả cao nhất.
3. **Debug bài bản**: mỗi lỗi thực nghiệm đều phải hiểu nguyên nhân và ghi lại cách khắc phục.

Trong toàn bộ quá trình, tôi sử dụng **Claude Code (Opus 4.7)** như một trợ lý code - thảo luận thiết kế, sinh boilerplate, rà lỗi. Các **quyết định kiến trúc, lựa chọn thuật toán, đánh giá kết quả** đều do tôi đưa ra dựa trên dữ liệu thực nghiệm.

---

## 3. Kiến trúc hệ thống

```
  Camera / Video / Ảnh đầu vào
              │
              ▼
  ┌────────────────────────────┐
  │  YOLOv8 Detection          │ ← mô hình tự train (v1/v2)
  │  (agnostic NMS, conf≥0.45) │
  └─────────────┬──────────────┘
                │ crop biển số
                ▼
  ┌────────────────────────────┐
  │  PlatePreprocessor         │  4 bước:
  │    • upscale (min 96px)    │    1. phóng to nếu nhỏ
  │    • CLAHE (tăng tương phản)│   2. chuẩn hóa ánh sáng
  │    • bilateral filter       │   3. khử nhiễu giữ cạnh
  │    • deskew (Hough lines)   │   4. cân chỉnh góc nghiêng
  └─────────────┬──────────────┘
                │
                ▼
  ┌────────────────────────────┐
  │  PaddleOCR v5              │ ← thay thế EasyOCR
  │  (en, mkldnn off)          │
  └─────────────┬──────────────┘
                │ text thô
                ▼
  ┌────────────────────────────┐
  │  correct_plate()           │ ← sửa nhầm theo vị trí
  │  is_valid() với regex VN   │    (O↔0, I↔1, B↔8, S↔5…)
  └─────────────┬──────────────┘
                │
                ▼ (chỉ video)
  ┌────────────────────────────┐
  │  PlateTracker              │ IoU matching +
  │  (votes_required=3)        │ majority voting qua 3 frame
  └─────────────┬──────────────┘
                │
                ▼
    Biển số cuối + log CSV
```

Mã nguồn core nằm trong [plate_pipeline.py](plate_pipeline.py) - gồm bốn module độc lập có thể test riêng.

---

## 4. Hành trình phát triển: tôi đã làm gì để đi từ 65% lên 95%

Đây là phần quan trọng nhất của dự án với tôi - **không phải kết quả cuối mà là quá trình đạt được nó**.

### Bước 1 - Xác định nút thắt

Sai lầm thường gặp: nghĩ rằng "chính xác thấp nghĩa là phải train lại mô hình". Tôi chọn cách khác: chạy detection một mình và đo mAP. Kết quả: **mAP50 = 99.5%** ngay với checkpoint đầu. Vậy nên nguyên nhân của 65% **không nằm ở detection**, mà ở khâu OCR và post-processing.

Đây là lần đầu tôi thấy rõ việc **instrument từng bước trong pipeline** quan trọng đến mức nào.

### Bước 2 - Thay OCR: EasyOCR → PaddleOCR

EasyOCR gặp ba vấn đề lớn trên biển số xe:
- Không có **allowlist** → đoán ra các ký tự không tồn tại trên biển VN (ví dụ `R`, `W`, `J`, `O`, `Q`).
- Không phân biệt tốt giữa số/chữ cái ở các vị trí khác nhau.
- Confidence không ổn định trên ảnh nhỏ/mờ.

Tôi chuyển sang **PaddleOCR v5** và thêm bộ lọc charset Việt Nam:
```python
VN_CHARSET = set("0123456789 ABCDEFGHKLMNPSTUVXYZ")
```

### Bước 3 — Preprocessing trước khi OCR

OCR chỉ tốt khi đầu vào rõ. Tôi thiết kế `PlatePreprocessor` với chuỗi 4 phép biến đổi:

| Bước | Kỹ thuật | Lý do |
|---|---|---|
| 1. Upscale | `cv2.resize` với `INTER_CUBIC` nếu chiều cao < 96px | PaddleOCR đọc tốt nhất ở height 64-96 |
| 2. CLAHE | `createCLAHE(clipLimit=3.0, tile 8×8)` | Chuẩn hóa tương phản cục bộ - xử lý ảnh sáng/tối không đều |
| 3. Bilateral filter | `d=7, sigma=50` | Khử nhiễu nhưng giữ cạnh ký tự sắc nét |
| 4. Deskew | Canny + HoughLinesP + median angle | Cân chỉnh biển bị nghiêng ±10° |

Ảnh kết quả được convert trở lại BGR để tương thích với input của PaddleOCR.

### Bước 4 - Validation & correction

OCR thô vẫn hay nhầm `O`↔`0`, `I`↔`1`, `B`↔`8`. Tôi không sửa bừa — **sửa theo vị trí** trên biển VN:

```python
# Biển VN có format: 2 số | 1-2 chữ + số phụ | 4-5 số
# Ví dụ hợp lệ: 30A-12345, 51F1-23456, 92H1-12345

PLATE_RE = r"^(\d{2})([A-Z]{1,2}\d??)(\d{4,5})$"
```

Hàm `correct_plate()` duyệt từng vị trí: 2 đầu phải là số, vị trí 3 phải là chữ, 4-5 đuôi phải là số - và sửa nhầm theo từ điển `_DIGIT_FIX` / `_LETTER_FIX`.

### Bước 5 - Multi-frame voting cho video

OCR từng frame độc lập dễ dao động. Tôi cài `PlateTracker`:
- Match các detection qua frame bằng **IoU** (threshold 0.35).
- Lưu lịch sử text đọc được vào `Counter` từng track.
- Khi track đọc được **≥3 lần nhất quán** và biển hợp lệ → "chốt" kết quả, log CSV.
- Track mất quá 20 frame → xóa.

Cơ chế này giúp độ chính xác trên video tăng từ ~85% (frame-level) lên ~95% (track-level).

---

## 5. So sánh 2 cách train: tại sao tôi chạy cả hai

### v1 — Train CPU local (baseline)

Tôi chạy trước trên chính máy của mình (AMD Ryzen 5 5500U, không GPU) để:
- Kiểm chứng pipeline hoạt động đúng end-to-end.
- Có **baseline** để so sánh với Colab.
- Tự tin rằng không phụ thuộc cloud.

**Config** (xem [train.py](train.py)):
```python
YOLOv8n, 30 epochs, batch=8, AdamW, cos_lr=True
patience=15 (early stop nếu val loss không giảm)
Augmentation:
  hsv_h=0.02, hsv_s=0.7, hsv_v=0.5 — robust sáng/tối
  degrees=10, translate=0.1, scale=0.5 — biến dạng hình học vừa phải
  fliplr=0.0 — BẮT BUỘC tắt, vì biển số lật ngang sẽ sai ký tự
  mosaic=1.0, mixup=0.1, copy_paste=0.1 — robust bối cảnh
```

Thời gian thực tế: **~6 giờ**. Loss giảm đều qua 30 epoch, không overfit.

### v2 - Train Colab GPU

Sau khi v1 xong và cho kết quả tốt, tôi chạy tiếp trên **Google Colab T4 GPU** với cấu hình mạnh hơn (xem [train_colab.ipynb](train_colab.ipynb)):
- Backbone **YOLOv8s** (lớn hơn v1's yolov8n 4 lần)
- **100 epochs** thay vì 30
- `batch=32`, augmentation mạnh hơn: `shear=2.0, perspective=0.0005, erasing=0.4`

Thời gian: **~1.5 giờ** trên T4 - nhanh hơn 4 lần so với CPU dù train 3 lần nhiều epoch hơn.

### Kết quả so sánh trực tiếp

| Tiêu chí | v1 CPU | v2 Colab GPU | Kết luận của tôi |
|---|---|---|---|
| mAP50 | 99.50% | ~99.5% | Ngang bằng — dataset đơn giản |
| mAP50-95 | 87.00% | 88-92% | v2 khá hơn ở IoU cao (bbox chính xác hơn) |
| Robustness thực tế | Tốt | Tốt hơn (có erasing, shear) | v2 thắng ở biển bị che khuất |
| Tốc độ inference CPU | 25 ms | 100 ms | **v1 nhanh gấp 4** cho edge device |
| Kích thước | 6.2 MB | 22.5 MB | v1 gọn hơn cho deploy |

**Quyết định của tôi**: dùng **v1 làm model production** cho webcam real-time (ưu tiên tốc độ). Dùng **v2 khi cần độ chính xác cao nhất** ở ảnh tĩnh hoặc video chất lượng cao.

Thực tế `demo.py` và `test.py` có hàm `resolve_default_model()` tự chọn model mới nhất theo mtime - nếu muốn ép dùng model cụ thể thì truyền `--model <path>`.

---

## 6. Nhật ký debug: các lỗi thực nghiệm và cách khắc phục

Tôi liệt kê ngắn gọn 11 vấn đề tôi gặp và giải quyết - mỗi mục đều là một bài học kỹ thuật.

| # | Vấn đề | Nguyên nhân gốc | Cách tôi sửa |
|---|---|---|---|
| 1 | EasyOCR đọc ra ký tự lạ (`R`, `W`) | Không có allowlist charset | Chuyển PaddleOCR v5 + filter `VN_CHARSET` |
| 2 | OCR fail trên biển mờ/tối | Không preprocessing | Thêm CLAHE + bilateral + deskew |
| 3 | Nhầm `O↔0`, `B↔8` không đoán được | Sửa bừa → gây nhầm mới | `correct_plate()` sửa theo vị trí kiểu số/chữ |
| 4 | Text rác vẫn được coi là biển | Không validate | Regex `PLATE_RE` kiểm tra đúng format VN |
| 5 | Mỗi frame ra khác nhau | Không memory giữa các frame | `PlateTracker` với IoU + majority voting |
| 6 | `test.py` trả kết quả kỳ lạ | Dùng `yolov8n.pt` pretrained COCO thay vì `best.pt` | Fix `MODEL_PATH` về file đã train |
| 7 | PaddleOCR crash `NotImplementedError: ConvertPirAttribute2RuntimeAttribute` | Bug oneDNN trên Windows CPU với Paddle 3.x | Truyền `enable_mkldnn=False, device="cpu"` khi init |
| 8 | Model Colab v2 sinh 2 bounding box chồng | Roboflow split 2 classes, NMS mặc định không gộp cross-class | Bật `agnostic_nms=True, iou=0.5` trong `detector.predict()` |
| 9 | Regex parse `"30E34422"` ra `30|E3|4422` | `\d?` greedy ưu tiên nhồi digit vào nhóm chữ | Đổi sang lazy `\d??` để ưu tiên phần số cuối dài 5 |
| 10 | Labels test có **BOM UTF-8** (`﻿`) → YOLO `could not convert string to float` | File được export lưu kèm BOM | Strip BOM trước khi training (v4 experimental) |
| 11 | `get_next_version()` tạo trùng thư mục | Chỉ quét `runs/detect/` nhưng thực tế path là `runs/detect/runs/` | Quét cả hai mức thư mục |

Mỗi fix trên đều là một PR nhỏ trong lịch sử git, có thể kiểm tra lại khi cần.

### Ví dụ điển hình

Ảnh test `Tgmt_0556_*.jpg` (biển thật `30E-34422`):

| Phiên bản pipeline | Detection | OCR đọc ra | Kết quả cuối |
|---|---|---|---|
| Ban đầu (EasyOCR) | 86% | `3OE3442Z` | Sai 3 ký tự |
| Thay PaddleOCR | 86% | `30E 344.22` | Join sai → sai format |
| + preprocessing | 86% | `30E` + `34422` | **30E-34422** ✓ |
| + agnostic_nms | 86% (1 box) | `30E34422` | `30E3-4422` (regex sai) |
| + lazy regex `\d??` | 86% | `30E34422` | **30E-34422** ✓ |

---

## 7. Cài đặt và chạy thử

### Yêu cầu
- Python ≥ 3.10
- Windows / Linux / macOS
- (Tùy chọn) GPU NVIDIA + CUDA cho training

### Cài đặt
```bash
git clone https://github.com/NghiaKitsune/vn-license-plate-yolov8.git
cd vn-license-plate-yolov8

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Sử dụng

```bash
# Webcam real-time (phím Q thoát, S chụp)
python demo.py

# Video file
python demo.py --source video.mp4

# Ảnh tĩnh
python demo.py --source anh_bien_so.jpg

# Test nhanh 1 ảnh với log chi tiết
python test.py --source anh_bien_so.jpg

# Chọn model cụ thể
python demo.py --model runs/detect/runs/bien_so_v2_gpu/weights/best.pt
```

### Tự train lại
```bash
# Train trên CPU (~6 giờ, yolov8n)
python train.py --mode cpu

# Train trên GPU (~1.5 giờ Colab T4, yolov8s)
python train.py --mode gpu

# Hoặc dùng notebook Colab:
# Upload train_colab.ipynb lên https://colab.research.google.com
# Runtime > Change runtime type > T4 GPU
```

---

## 8. Cấu trúc dự án

```
vn-license-plate-yolov8/
├── README.md                     Tài liệu này
├── requirements.txt              Dependencies
├── .gitignore
│
├── plate_pipeline.py             Core: preprocessor, OCR, validator, tracker
├── demo.py                       CLI webcam/video/ảnh
├── test.py                       Test nhanh 1 ảnh
├── train.py                      Training CPU/GPU + resume
├── train_colab.ipynb             Notebook Colab GPU
│
├── dataset/
│   └── data.yaml                 Cấu hình YOLO (ảnh tải từ Roboflow)
│
└── runs/
    └── detect/runs/
        ├── bien_so_v1/           Model v1 (CPU, yolov8n)
        │   ├── weights/best.pt   ← model mặc định
        │   ├── results.csv       Log training 30 epoch
        │   ├── results.png       Biểu đồ loss + mAP
        │   └── confusion_matrix.png
        └── bien_so_v2_gpu/       Model v2 (Colab, yolov8s) — chỉ weights
```

---

## 9. Công nghệ sử dụng

- **Python 3.10**
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** 8.4.39 - detection
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** 3.0+ - OCR
- **OpenCV 4.10** - image processing (CLAHE, bilateral, Hough, warpAffine)
- **NumPy 1.26**
- **[Roboflow](https://roboflow.com)** - hosting dataset
- **Google Colab** (T4 GPU) — training v2

### Dataset
[Vietnam License Plate - hjswj v2](https://universe.roboflow.com/vietnam-license/vietnam-license-plate-hjswj/dataset/2) trên Roboflow Universe, CC BY 4.0. Gồm 4072 ảnh (2850 train / 815 valid / 407 test).

### Charset của biển số VN
```
Số:   0 1 2 3 4 5 6 7 8 9
Chữ:  A B C D E F G H K L M N P S T U V X Y Z
```
Biển số VN **không dùng** các chữ `I, J, O, Q, R, W` do dễ nhầm với số. Đây là thông tin quan trọng tôi áp dụng vào `VN_CHARSET` và `PLATE_RE`.

---

## 10. Hạn chế hiện tại và hướng phát triển

Điểm tôi tự đánh giá là còn cải thiện được:

- **Biển 2 dòng (xe máy)**: PaddleOCR đôi khi đọc ngược thứ tự dòng. Cần tách crop thành 2 nửa trên/dưới và OCR riêng rồi ghép.
- **Biển bị che khuất một phần**: Augmentation `erasing=0.4` ở v2 có giúp nhưng chưa đủ. Có thể thử **Copy-Paste augmentation** với patch occlusion.
- **Biển nước ngoài / biển đặc biệt** (ngoại giao, quân đội): Pipeline hiện mặc định format dân sự. Cần thêm regex riêng.
- **Deploy edge device**: Model `yolov8n` đủ nhẹ nhưng PaddleOCR vẫn nặng cho Raspberry Pi. Có thể thử quantize hoặc thay bằng TrOCR-tiny.
- **Confidence calibration**: Hiện score OCR của PaddleOCR không được calibrate với ground truth thực tế. Có thể fit một logistic regression để ra confidence "thật".

Hướng mở rộng tiếp theo tôi muốn làm:
1. **Web UI với Streamlit** để demo online.
2. **Export sang ONNX + TensorRT** để đo tốc độ trên Jetson Nano.
3. **Fine-tune trên video CCTV** thật để xử lý motion blur + low light đêm.

---

## 11. Lời kết

Dự án này không phải chỉ để có một con số mAP đẹp - **giá trị lớn nhất với tôi là quá trình gỡ lỗi và ra quyết định kỹ thuật** ở từng bước. Mỗi lỗi gặp phải, từ bug oneDNN trên Windows đến vấn đề regex greedy, đều dạy tôi một khía cạnh mới về pipeline thị giác máy tính thực tế.

Tôi chia sẻ repo này với hy vọng nó có ích cho người khác đang học CV/OCR, đặc biệt là các bạn sinh viên Việt Nam muốn thực hành trên bài toán gần gũi với cuộc sống.

**Tác giả**: Nghia Vo - nghiavo201106@gmail.com
**GitHub**: https://github.com/NghiaKitsune/vn-license-plate-yolov8
**Hỗ trợ code**: Claude Code (Anthropic) - dùng như trợ lý viết code, không thay thế vai trò thiết kế.

---

*License: MIT (code). Dataset theo giấy phép CC BY 4.0 của Roboflow workspace vietnam-license.*
