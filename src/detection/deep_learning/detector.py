"""
================================================================================
PHƯƠNG PHÁP 2 (THÀNH VIÊN 2 PHỤ TRÁCH): DEEP LEARNING DETECTOR / CLASSIFIER
Kiến trúc chính: Mạng Nơ-ron Tích Chập LeNet-5 Cải Tiến (Enhanced LeNet-5).
(Đồng thời hỗ trợ linh hoạt nạp checkpoint YOLOv8 nếu có).
================================================================================
"""

from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

from src.config import (
    TARGET_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    CHECKPOINTS_DIR,
)
from src.detection.base_detector import BaseDetector


class DeepLearningDetector(BaseDetector):
    """
    Lớp suy luận Deep Learning cho Thành viên 2 (Enhanced LeNet / CNN).
    Kế thừa BaseDetector để tương thích 100% với hệ thống Benchmark và App.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="deep_learning", checkpoint_path=checkpoint_path)
        self.arch_type = "unknown"
        self.device = "cpu"
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Nạp trọng số mô hình từ tệp checkpoint (.pth, .pt).
        Tự động ưu tiên nạp mô hình LeNet-5 Cải Tiến (PyTorch).
        """
        if checkpoint_path is None:
            default_dir = CHECKPOINTS_DIR / "deep_learning"
            cnn_ckpt = default_dir / "best_cnn.pth"
            yolo_ckpt = default_dir / "best_yolo.pt"

            if cnn_ckpt.exists():
                checkpoint_path = str(cnn_ckpt)
            elif yolo_ckpt.exists():
                checkpoint_path = str(yolo_ckpt)

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self.model = None
            return

        self.checkpoint_path = checkpoint_path
        ckpt_path = Path(checkpoint_path)

        # 1. Trường hợp nạp mô hình PyTorch CNN (Enhanced LeNet)
        if ckpt_path.suffix.lower() in [".pth", ".pt"] and "yolo" not in ckpt_path.name.lower():
            try:
                import torch
                from src.detection.deep_learning.lenet_model import EnhancedLeNet

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model = EnhancedLeNet(num_classes=len(TARGET_CLASSES))

                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    val_acc = checkpoint.get("val_acc", 0.0)
                    print(f"[+] [DeepLearning] Đã nạp Enhanced LeNet (Val Acc: {val_acc*100:.2f}%) từ: {checkpoint_path}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"[+] [DeepLearning] Đã nạp Enhanced LeNet từ: {checkpoint_path}")

                self.model = model.to(self.device).eval()
                self.arch_type = "cnn"
                return
            except Exception as e:
                print(f"[-] Lỗi nạp checkpoint CNN: {e}")
                self.model = None

        # 2. Trường hợp nạp mô hình YOLOv8/v11
        if "yolo" in ckpt_path.name.lower():
            try:
                from ultralytics import YOLO
                self.model = YOLO(checkpoint_path)
                self.arch_type = "yolo"
                print(f"[+] [DeepLearning] Đã nạp thành công mô hình YOLO từ: {checkpoint_path}")
                return
            except ImportError:
                print("[-] Vui lòng cài đặt ultralytics để dùng YOLO: pip install ultralytics")
            except Exception as e:
                print(f"[-] Lỗi nạp checkpoint YOLO: {e}")

    def preprocess(self, image: np.ndarray) -> Any:
        """
        Tiền xử lý ảnh đầu vào BGR.
        Với LeNet: Resize về IMAGE_SIZE (64x64), chuẩn hóa ImageNet, chuyển đổi Tensor.
        """
        if self.model is None:
            raise NotImplementedError(
                "Mô hình Deep Learning chưa được huấn luyện hoặc chưa nạp checkpoint!\n"
                "-> Hãy chạy lệnh huấn luyện Enhanced LeNet:\n"
                "   python -m src.detection.deep_learning.train --arch cnn --epochs 15"
            )

        if self.arch_type == "cnn":
            resized = cv2.resize(image, IMAGE_SIZE)
            # Chuyển BGR sang RGB và chuẩn hóa theo phân phối ImageNet
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb - mean) / std

            # Chuyển format HWC -> CHW cho PyTorch
            tensor_data = np.transpose(normalized, (2, 0, 1))
            return tensor_data
        else:
            return image

    def inference(self, preprocessed_input: Any) -> Any:
        """Thực thi dự đoán mô hình."""
        if self.arch_type == "cnn":
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(preprocessed_input).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                return logits
        elif self.arch_type == "yolo":
            results = self.model.predict(preprocessed_input, verbose=False)
            return results
        else:
            raise NotImplementedError("Kiến trúc mô hình chưa xác định.")

    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """
        Hậu xử lý kết quả dự đoán trả về dict chuẩn:
        {
            "class_name": str,
            "class_id": int,
            "confidence": float,
            "bbox": tuple | None,
        }
        """
        if self.arch_type == "cnn":
            import torch
            probs = torch.softmax(raw_output, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            conf = float(probs[pred_id])
            pred_class = IDX_TO_CLASS.get(pred_id, "Unknown")
            return {
                "class_name": pred_class,
                "class_id": pred_id,
                "confidence": conf,
                "bbox": None,
            }
        elif self.arch_type == "yolo":
            boxes = raw_output[0].boxes
            if len(boxes) == 0:
                return {
                    "class_name": "Unknown",
                    "class_id": -1,
                    "confidence": 0.0,
                    "bbox": None,
                }
            best_idx = int(boxes.conf.argmax().cpu().item())
            cls_id = int(boxes.cls[best_idx].cpu().item())
            conf = float(boxes.conf[best_idx].cpu().item())
            xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
            pred_class = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else "Unknown"

            return {
                "class_name": pred_class,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": tuple(xyxy),
            }
        else:
            return {
                "class_name": "Unknown",
                "class_id": -1,
                "confidence": 0.0,
                "bbox": None,
            }
