"""
Pipeline nhan dien bien so xe Viet Nam.
Robust voi anh sang/nhieu/chuyen dong da dang.

Modules:
  PlatePreprocessor : CLAHE + bilateral + upscale + deskew
  PlateOCR          : PaddleOCR + allowlist + confusion correction
  PlateValidator    : regex format bien VN + sua loi ky tu
  PlateTracker      : multi-frame IoU tracking + majority voting
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from paddleocr import PaddleOCR


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

OCR_CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VN_CHARSET = set("0123456789ABCDEFGHKLMNPSTUVXYZ")
MIN_PLATE_LEN = 7
MAX_PLATE_LEN = 10


class PlatePreprocessor:
    """Chuan hoa crop bien so truoc khi OCR."""

    def __init__(self, target_height: int = 96) -> None:
        self.target_height = target_height
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        variants = self.variants(crop)
        return variants[0] if variants else crop

    def variants(self, crop: np.ndarray) -> list[np.ndarray]:
        if crop is None or crop.size == 0:
            return []
        gray = self._prepare_gray(crop)
        sharp = self._sharpen(gray)
        otsu = cv2.threshold(
            sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        adaptive = cv2.adaptiveThreshold(
            sharp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return [
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for img in (sharp, gray, otsu, adaptive)
        ]

    def _prepare_gray(self, crop: np.ndarray) -> np.ndarray:
        crop = self._upscale(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        return self._deskew(gray)

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.target_height:
            scale = self.target_height / h
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        return img

    @staticmethod
    def _sharpen(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=gray.shape[1] // 4, maxLineGap=20,
        )
        if lines is None:
            return gray
        angles = [
            np.degrees(np.arctan2(y2 - y1, x2 - x1))
            for x1, y1, x2, y2 in (ln[0] for ln in lines[:30])
            if x2 != x1 and abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 20
        ]
        if not angles or abs(np.median(angles)) < 0.5:
            return gray
        angle = float(np.median(angles))
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 2. OCR
# ---------------------------------------------------------------------------

class PlateOCR:
    """PaddleOCR wrapper loc theo charset bien VN."""

    def __init__(self) -> None:
        # enable_mkldnn=False: tranh bug oneDNN tren Windows CPU
        # (NotImplementedError ConvertPirAttribute2RuntimeAttribute)
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
            device="cpu",
        )

    def read(self, img: np.ndarray) -> tuple[str, float]:
        """Tra ve (text_sach, confidence). '' neu khong doc duoc."""
        if img is None or img.size == 0:
            return "", 0.0
        try:
            result = self._ocr.predict(img)
        except Exception:
            return "", 0.0
        if not result:
            return "", 0.0

        texts: list[str] = []
        scores: list[float] = []
        for page in result:
            data = page if isinstance(page, dict) else {}
            if not data and hasattr(page, "json"):
                data = page.json.get("res", {})
            texts.extend(data.get("rec_texts") or [])
            scores.extend(data.get("rec_scores") or [])

        if not texts:
            return "", 0.0

        merged = "".join(texts).upper()
        cleaned = "".join(c for c in merged if c in OCR_CHARSET)
        avg_conf = float(np.mean(scores)) if scores else 0.0
        return cleaned, avg_conf

    def read_best(self, imgs: list[np.ndarray]) -> tuple[str, float]:
        """Thu nhieu bien the preprocess va chon ket qua OCR hop le nhat."""
        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for img in imgs:
            raw_text, raw_conf = self.read(img)
            normalized = normalize_plate(raw_text)
            score = candidate_score(normalized, raw_conf)
            if score > best_score:
                best_text = normalized
                best_conf = raw_conf
                best_score = score
            if normalized and is_valid(normalized) and raw_conf >= 0.8:
                break

        return best_text, best_conf


# ---------------------------------------------------------------------------
# 3. Validation + correction
# ---------------------------------------------------------------------------

# Format bien VN: 2 so + 1-2 chu (+ so phu) + 4-5 so
# Vi du: 30A12345, 51F1-23456, 92H112345
# Lazy `\d?` trong nhom chu: uu tien dai so cuoi 5 chu so hon 4 chu so
# Vi du: "30E34422" -> "30|E|34422" thay vi "30|E3|4422"
PLATE_RE = re.compile(r"^(\d{2})([A-Z]{1,2}\d??)(\d{4,5})$")

_DIGIT_FIX  = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
               "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
_LETTER_FIX = {"0": "D", "1": "I", "2": "Z", "5": "S",
               "8": "B", "6": "G", "7": "T"}


def correct_plate(text: str) -> str:
    """Ep ky tu vao dung loai (so/chu) theo vi tri tren bien VN."""
    if not MIN_PLATE_LEN <= len(text) <= MAX_PLATE_LEN:
        return text
    chars = list(text)
    for i in range(2):                          # 2 dau: phai la so
        if not chars[i].isdigit():
            chars[i] = _DIGIT_FIX.get(chars[i], chars[i])
    if not chars[2].isalpha():                  # vi tri 3: phai la chu
        chars[2] = _LETTER_FIX.get(chars[2], chars[2])
    n = len(chars)
    for i in range(n - 1, max(n - 6, 2), -1):  # 4-5 cuoi: phai la so
        if not chars[i].isdigit():
            chars[i] = _DIGIT_FIX.get(chars[i], chars[i])
    return "".join(chars)


def is_valid(text: str) -> bool:
    return bool(PLATE_RE.match(text))


def format_plate(text: str) -> str:
    """30A12345 -> '30A-12345'"""
    m = PLATE_RE.match(text)
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}" if m else text


def candidate_score(text: str, conf: float) -> float:
    if not text:
        return -1.0
    score = conf
    if MIN_PLATE_LEN <= len(text) <= MAX_PLATE_LEN:
        score += 0.4
    if len(text) >= 2 and text[:2].isdigit():
        score += 0.2
    if text[-4:].isdigit():
        score += 0.3
    if is_valid(text):
        score += 2.0
    return score


def normalize_plate(text: str) -> str:
    """Lam sach OCR va chon candidate hop le nhat theo format bien VN."""
    cleaned = "".join(c for c in text.upper() if c in OCR_CHARSET)
    if not cleaned:
        return ""

    candidates: list[str] = []
    seen_windows: set[str] = set()
    for size in range(min(MAX_PLATE_LEN, len(cleaned)), MIN_PLATE_LEN - 1, -1):
        for start in range(0, len(cleaned) - size + 1):
            window = cleaned[start:start + size]
            if window in seen_windows:
                continue
            seen_windows.add(window)
            candidates.append(correct_plate(window))

    if not candidates:
        return correct_plate(cleaned)

    best = max(candidates, key=lambda item: candidate_score(item, 0.0))
    return best


# ---------------------------------------------------------------------------
# 4. Multi-frame tracker
# ---------------------------------------------------------------------------

Box = tuple[int, int, int, int]  # x1,y1,x2,y2


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    box: Box
    reads: Counter = field(default_factory=Counter)
    last_seen: int = 0
    logged: bool = False

    @property
    def best(self) -> tuple[str, float]:
        if not self.reads:
            return "", 0.0
        return self.reads.most_common(1)[0]


class PlateTracker:
    """Track bien qua nhieu frame, lay ket qua OCR chiem da so.

    Args:
        iou_threshold:  IoU toi thieu de coi la cung mot bien.
        votes_required: So lan doc nhat quan de 'chot' bien.
        max_missed:     Frame lien tiep mat bien roi xoa track.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        votes_required: int = 3,
        max_missed: int = 20,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.votes_required = votes_required
        self.max_missed = max_missed
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._frame = 0

    def update(
        self, detections: list[tuple[Box, str, float]]
    ) -> list[tuple[int, Box, str, bool]]:
        """
        detections: [(box, ocr_text, vote_weight), ...]
        Returns: [(track_id, box, best_text, is_confirmed), ...]
        """
        self._frame += 1
        results: list[tuple[int, Box, str, bool]] = []
        matched: set[int] = set()

        for box, text, vote_weight in detections:
            best_id, best_score = None, 0.0
            for tid, tr in self._tracks.items():
                if tid in matched:
                    continue
                score = _iou(box, tr.box)
                if score > best_score:
                    best_id, best_score = tid, score

            if best_id is not None and best_score >= self.iou_threshold:
                tr = self._tracks[best_id]
                tr.box = box
                tr.last_seen = self._frame
                if text and vote_weight > 0:
                    tr.reads[text] += vote_weight
                matched.add(best_id)
                tid = best_id
            else:
                tid = self._next_id
                self._next_id += 1
                tr = _Track(box=box, last_seen=self._frame)
                if text and vote_weight > 0:
                    tr.reads[text] += vote_weight
                self._tracks[tid] = tr

            best_text, score = tr.best
            confirmed = score >= self.votes_required and is_valid(best_text)
            results.append((tid, box, best_text, confirmed))

        # Don rac track cu
        stale = [tid for tid, tr in self._tracks.items()
                 if self._frame - tr.last_seen > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

        return results

    def mark_logged(self, track_id: int) -> None:
        if track_id in self._tracks:
            self._tracks[track_id].logged = True

    def is_logged(self, track_id: int) -> bool:
        return track_id in self._tracks and self._tracks[track_id].logged


# ---------------------------------------------------------------------------
# 5. Full inference helper (anh tinh / webcam frame)
# ---------------------------------------------------------------------------

def run_on_frame(
    frame: np.ndarray,
    detector,
    preprocessor: PlatePreprocessor,
    ocr: PlateOCR,
    conf_threshold: float = 0.45,
) -> list[tuple[Box, str, float]]:
    """
    Chay detection + OCR tren mot frame.
    Tra ve [(box, plate_text, det_conf, ocr_conf), ...]
    """
    # agnostic_nms=True: gop box trung lap giua cac class
    # (model v2_gpu co 2 classes cung la bien so, NMS mac dinh khong suppress cross-class)
    results = detector.predict(
        frame, conf=conf_threshold, verbose=False,
        agnostic_nms=True, iou=0.5,
    )
    detections: list[tuple[Box, str, float, float]] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            det_conf = float(box.conf[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            processed_variants = preprocessor.variants(crop)
            plate_text, ocr_conf = ocr.read_best(processed_variants)
            if ocr_conf < 0.25 and not is_valid(plate_text):
                plate_text = ""
            detections.append(((x1, y1, x2, y2), plate_text, det_conf, ocr_conf))
    return detections
