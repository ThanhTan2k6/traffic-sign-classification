"""
================================================================================
PHƯƠNG PHÁP 1: MÔ HÌNH HOG + COLOR HISTOGRAM + SVM (THÀNH VIÊN 1)
================================================================================
Kế thừa BaseDetector để tương thích hoàn toàn với hệ thống Benchmark và CLI demo.
- Preprocess : Trích xuất đặc trưng Color + HOG (1796 chiều) từ ảnh ROI BGR
- Inference  : Dự đoán xác suất với SVM Pipeline (StandardScaler + Calibrated LinearSVC)
- Postprocess: Trả về dict định dạng chuẩn:
  {"class_name": str, "class_id": int, "confidence": float, "bbox": tuple | None}
================================================================================
"""

from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple
import cv2
import joblib
import numpy as np

from src.config import CHECKPOINTS_DIR, TARGET_CLASSES, CLASS_TO_IDX
from src.detection.base_detector import BaseDetector
from src.detection.color_hog_svm.features import extract_color_hog_features


class ColorHogSvmDetector(BaseDetector):
    """Lớp suy luận cho Phương pháp 1: Color + HOG + SVM (Thành viên 1)."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="color_hog_svm", checkpoint_path=checkpoint_path)
        if checkpoint_path is None:
            default_ckpt = CHECKPOINTS_DIR / "color_hog_svm" / "best_svm.pkl"
            if default_ckpt.exists():
                checkpoint_path = str(default_ckpt)

        self.pipeline = None
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Nạp pipeline tiền xử lý và mô hình SVM từ tệp checkpoint .pkl."""
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self.model = None
            self.pipeline = None
            return

        self.checkpoint_path = checkpoint_path
        try:
            ckpt_data = joblib.load(checkpoint_path)
            if isinstance(ckpt_data, dict) and "pipeline" in ckpt_data:
                self.pipeline = ckpt_data["pipeline"]
                self.model = self.pipeline
                val_acc = ckpt_data.get("val_acc", 0.0)
                print(f"[+] [ColorHogSVM] Đã nạp thành công mô hình SVM (Val Acc: {val_acc*100:.2f}%) từ: {checkpoint_path}")
            else:
                self.pipeline = ckpt_data
                self.model = self.pipeline
                print(f"[+] [ColorHogSVM] Đã nạp thành công pipeline SVM từ: {checkpoint_path}")
        except Exception as e:
            print(f"[-] Lỗi khi nạp checkpoint SVM: {e}")
            self.pipeline = None
            self.model = None

    def localize_sign_roi(
        self, image: np.ndarray, is_roi: Optional[bool] = None
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Định vị vùng chứa biển báo (ROI) từ ảnh chụp toàn cảnh đường phố
        sử dụng bộ lọc màu HSV và phép toán hình thái học (Morphology).
        - Nếu is_roi=True: Bỏ qua bước định vị và dùng trực tiếp ảnh đầu vào.
        - Nếu is_roi=False hoặc None: Quét màu HSV và cắt ROI nếu tìm thấy biển báo.
        """
        if is_roi is True:
            return image, None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Mặt nạ màu Đỏ (Biển Cấm, Biển Nguy Hiểm)
        m_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        m_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        # Mặt nạ màu Xanh Lam (Biển Hiệu Lệnh, Chỉ Dẫn)
        m_blue = cv2.inRange(hsv, (100, 70, 50), (130, 255, 255))
        # Mặt nạ màu Vàng (Biển Nguy Hiểm)
        m_yellow = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))

        mask = m_red1 | m_red2 | m_blue | m_yellow
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_area = 0

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            aspect_ratio = bw / max(1.0, bh)
            # Biển báo hợp lệ có tỉ lệ w/h xấp xỉ 0.4 đến 2.5
            if 15 <= bw <= 600 and 15 <= bh <= 600 and 0.4 <= aspect_ratio <= 2.5:
                if area > best_area:
                    best_area = area
                    best_box = (x, y, x + bw, y + bh)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                return crop, best_box

        # Nếu không định vị được contour biển báo rõ ràng, giữ nguyên ảnh đầu vào
        return image, None

    def preprocess(self, image: np.ndarray, is_roi: bool = True) -> np.ndarray:
        """
        Tiền xử lý ảnh đầu vào BGR:
        - Mặc định (is_roi=True): Xử lý trực tiếp ảnh ROI biển báo cắt sẵn (phục vụ Benchmark).
        - Nếu is_roi=False: Định vị và cắt vùng ROI biển báo qua HSV + Morphology (ảnh toàn cảnh đường phố).
        - Trích xuất vector đặc trưng Color HSV Hist + HOG (1796 chiều).
        """
        if self.pipeline is None:
            raise NotImplementedError("Mô hình HOG+SVM chưa được nạp checkpoint. Hãy huấn luyện trước!")

        if is_roi:
            roi_img = image
            bbox = None
        else:
            roi_img, bbox = self.localize_sign_roi(image, is_roi=False)
        self._last_bbox = bbox

        feat = extract_color_hog_features(roi_img)
        return np.expand_dims(feat, axis=0)

    def inference(self, preprocessed_input: np.ndarray) -> np.ndarray:
        """Thực thi dự đoán xác suất với pipeline SVM và căn chỉnh với TARGET_CLASSES."""
        probs = np.zeros(len(TARGET_CLASSES), dtype=np.float32)

        # Lấy danh sách các lớp đã được huấn luyện
        fitted_classes = None
        if hasattr(self.pipeline, "classes_"):
            fitted_classes = self.pipeline.classes_
        elif hasattr(self.pipeline, "named_steps") and hasattr(self.pipeline.named_steps.get("classifier"), "classes_"):
            fitted_classes = self.pipeline.named_steps["classifier"].classes_

        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(preprocessed_input)[0]
            if fitted_classes is not None and len(fitted_classes) == len(proba):
                for label, p_val in zip(fitted_classes, proba):
                    idx = CLASS_TO_IDX.get(label, label) if isinstance(label, str) else int(label)
                    if 0 <= idx < len(TARGET_CLASSES):
                        probs[idx] = p_val
            else:
                probs[:len(proba)] = proba
            return probs
        else:
            raise RuntimeError(
                "Mô hình SVM yêu cầu CalibratedClassifierCV để sinh xác suất (calibrated probabilities). "
                "Hãy huấn luyện lại bằng 'python -m src.detection.color_hog_svm.train'."
            )

    def postprocess(self, raw_output: np.ndarray) -> Dict[str, Any]:
        """
        Hậu xử lý kết quả dự đoán trả về từ inference.
        Trả về dict chuẩn:
        {"class_name": str, "class_id": int, "confidence": float, "bbox": None}
        """
        class_id = int(np.argmax(raw_output))
        confidence = float(raw_output[class_id])
        class_name = TARGET_CLASSES[class_id]

        return {
            "class_name": class_name,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bbox": getattr(self, "_last_bbox", None),
        }

    def predict(self, image: np.ndarray, is_roi: bool = True) -> Dict[str, Any]:
        """
        Dự đoán với mô hình HOG + SVM:
        - is_roi=True (mặc định): Phân loại trực tiếp ảnh ROI biển báo (phục vụ Benchmark).
        - is_roi=False: Định vị vùng biển báo (HSV + Morphology) trên ảnh toàn cảnh trước khi phân loại.
        """
        if not is_roi:
            return self.predict_scene(image)
        return super().predict(image, is_roi=True)

    def predict_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Đường dẫn dành riêng cho ảnh chụp toàn cảnh đường phố:
        - Định vị vùng biển báo (HSV + Morphology).
        - Nếu không phát hiện biển báo, trả về kết quả Unknown (class_id: -1, confidence: 0.0, bbox: None).
        - Nếu phát hiện, cắt ROI và đưa vào mô hình SVM phân loại.
        """
        start_time = time.perf_counter()
        if self.pipeline is None:
            raise NotImplementedError("Mô hình HOG+SVM chưa được nạp checkpoint. Hãy huấn luyện trước!")
        roi_img, bbox = self.localize_sign_roi(image, is_roi=False)
        if bbox is None:
            latency = time.perf_counter() - start_time
            return {
                "class_name": "Unknown",
                "class_id": -1,
                "confidence": 0.0,
                "bbox": None,
                "inference_time": latency,
                "fps": (1.0 / latency) if latency > 0 else 0.0,
                "model_name": self.model_name,
            }

        self._last_bbox = bbox
        feat = extract_color_hog_features(roi_img)
        preprocessed = np.expand_dims(feat, axis=0)
        raw_output = self.inference(preprocessed)
        res = self.postprocess(raw_output)

        latency = time.perf_counter() - start_time
        res["inference_time"] = latency
        res["fps"] = (1.0 / latency) if latency > 0 else 0.0
        res["model_name"] = self.model_name
        return res
