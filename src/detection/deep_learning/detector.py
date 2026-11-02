"""
================================================================================
PHƯƠNG PHÁP 2 (THÀNH VIÊN 2 PHỤ TRÁCH): DEEP LEARNING DETECTOR / CLASSIFIER
Hỗ trợ linh hoạt tùy chọn giữa 2 kiến trúc theo sở thích của thành viên:
- Lựa chọn A: Mạng nơ-ron tích chập (Custom CNN / LeNet-5 / MobileNet)
- Lựa chọn B: Mô hình phát hiện đối tượng End-to-End (YOLOv8 / YOLOv11)
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
    Lớp suy luận Deep Learning cho Thành viên 2.
    Kế thừa BaseDetector để tương thích 100% với hệ thống Benchmark và App.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="deep_learning", checkpoint_path=checkpoint_path)
        self.arch_type = "unknown"  # 'cnn' hoặc 'yolo'
        self.device = "cpu"
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Nạp trọng số mô hình từ tệp checkpoint (.pth, .pt).
        Tự động nhận diện mô hình CNN (PyTorch) hoặc YOLO (Ultralytics).
        """
        if checkpoint_path is None:
            # Tìm checkpoint mặc định trong thư mục checkpoints/deep_learning/
            default_dir = CHECKPOINTS_DIR / "deep_learning"
            cnn_ckpt = default_dir / "best_cnn.pth"
            yolo_ckpt = default_dir / "best_yolo.pt"

            if cnn_ckpt.exists():
                checkpoint_path = str(cnn_ckpt)
            elif yolo_ckpt.exists():
                checkpoint_path = str(yolo_ckpt)

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            # Chưa có checkpoint được train
            self.model = None
            return

        self.checkpoint_path = checkpoint_path
        ckpt_path = Path(checkpoint_path)

        # 1. Trường hợp nạp mô hình YOLOv8/v11
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

        # 2. Trường hợp nạp mô hình PyTorch CNN
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # TODO: [Thành viên 2] Khởi tạo class kiến trúc mạng CNN và load state_dict
            # checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # self.model = CustomCNN(num_classes=len(TARGET_CLASSES))
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model.to(self.device).eval()
            self.arch_type = "cnn"
            print(f"[+] [DeepLearning] Đã nhận diện checkpoint CNN: {checkpoint_path}")
        except Exception as e:
            print(f"[-] Lỗi nạp checkpoint CNN: {e}")

    def preprocess(self, image: np.ndarray) -> Any:
        """
        Tiền xử lý ảnh đầu vào BGR.
        - Với CNN: Resize về IMAGE_SIZE (64x64), chuẩn hóa [0, 1], chuyển đổi Tensor.
        - Với YOLO: Đưa thẳng ảnh BGR vào để mô hình tự xử lý letterbox.
        """
        if self.model is None:
            raise NotImplementedError(
                "Mô hình Deep Learning chưa được huấn luyện hoặc chưa nạp checkpoint!\n"
                "-> Thành viên 2 hãy chạy huấn luyện bằng lệnh:\n"
                "   python -m src.detection.deep_learning.train --arch cnn   (hoặc --arch yolo)"
            )

        if self.arch_type == "cnn":
            resized = cv2.resize(image, IMAGE_SIZE)
            # Chuyển BGR -> RGB và chuẩn hóa
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Chuyển HWC -> CHW
            tensor_data = np.transpose(rgb, (2, 0, 1))
            return tensor_data
        else:
            # Với YOLO chỉ cần ảnh numpy BGR
            return image

    def inference(self, preprocessed_input: Any) -> Any:
        """Thực thi suy luận mô hình."""
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
            # Xử lý kết quả từ YOLO Results object
            boxes = raw_output[0].boxes
            if len(boxes) == 0:
                return {
                    "class_name": "Unknown",
                    "class_id": -1,
                    "confidence": 0.0,
                    "bbox": None,
                }
            # Lấy box có độ tin cậy cao nhất
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
