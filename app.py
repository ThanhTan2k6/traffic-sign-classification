"""
================================================================================
🚦 CLI ĐIỀU PHỐI TRUNG TÂM (CENTRAL PIPELINE RUNNER)
Đồ án Thị Giác Máy Tính: Nhận Dạng Biển Báo Giao Thông Việt Nam (QCVN 41:2019)
================================================================================
Hướng dẫn sử dụng nhanh qua CLI:

1. Chạy Benchmark phân loại (Classification) trên toàn bộ tập Test:
   python app.py --mode evaluate --split test

2. Chạy Benchmark đối chuẩn toàn trình trên ảnh đường phố toàn cảnh (End-to-End):
   python app.py --mode evaluate-scene --split test

3. Chạy đánh giá từng mô hình riêng lẻ:
   python app.py --mode evaluate --model color_hog_svm --split test
   python app.py --mode evaluate --model deep_learning --split test

4. Chạy Demo trên 1 ảnh biển báo (ROI crop):
   python app.py --mode demo --model deep_learning --input data/cropped/test/Bien_Cam/0002_00.jpg
   python app.py --mode demo --model color_hog_svm --input data/cropped/test/Bien_Cam/0002_00.jpg

5. Chạy Demo trên ảnh toàn cảnh đường phố (Tự động phát hiện bằng YOLO11s):
   python app.py --mode demo --model deep_learning --input dataset/images/0001.jpg --scene
================================================================================
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Cấu hình encoding UTF-8 cho Windows console
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Đảm bảo đường dẫn gốc được nhận diện
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SUPPORTED_MODELS,
    TEST_SPLIT_FILE,
    VAL_SPLIT_FILE,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    TARGET_CLASSES,
)
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.evaluation.eval_metrics import evaluate_detector, benchmark_all_models
from src.evaluation.plotter import plot_benchmark_comparison, plot_confusion_matrix
from src.evaluation.scene_eval import run_scene_benchmark, load_yolo_detector
from src.utils import imread_unicode, ensure_dir


def get_detector(model_name: str):
    """Factory helper khởi tạo detector theo tên."""
    name = (model_name or "deep_learning").lower()
    if name == "color_hog_svm":
        return ColorHogSvmDetector()
    elif name == "deep_learning":
        return DeepLearningDetector()
    else:
        raise ValueError(
            f"Mô hình '{model_name}' không được hỗ trợ. "
            f"Danh sách mô hình hợp lệ: {SUPPORTED_MODELS}"
        )


def run_evaluation(model_name: str = None, split_name: str = "test", max_samples: int = None) -> int:
    """Chế độ Đánh giá & So sánh (Evaluation / Benchmark Mode)."""
    split_file = TEST_SPLIT_FILE if split_name.lower() == "test" else VAL_SPLIT_FILE

    if not split_file.exists():
        print(f"[-] [LỖI] Không tìm thấy file split: {split_file}")
        return 1

    # Trường hợp 1: Chạy so sánh TOÀN BỘ các mô hình khi không chỉ định --model
    if model_name is None:
        print("\n🚀 Không chỉ định --model cụ thể. Khởi chạy Đối chuẩn TOÀN DIỆN tất cả mô hình...")
        detectors = [
            ColorHogSvmDetector(),
            DeepLearningDetector(),
        ]
        df_leaderboard = benchmark_all_models(detectors, split_file, max_samples=max_samples)
        if df_leaderboard.empty:
            print("[-] [LỖI] Không có mô hình nào chạy thành công.")
            return 1

        ensure_dir(RESULTS_DIR)
        out_chart = RESULTS_DIR / "benchmark_comparison.png"
        plot_benchmark_comparison(df_leaderboard, out_chart)
        return 0

    # Trường hợp 2: Chạy đánh giá 1 mô hình cụ thể
    try:
        detector = get_detector(model_name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"[-] [LỖI] {e}")
        return 1

    metrics = evaluate_detector(detector, split_file, max_samples=max_samples)
    if not metrics:
        print("[-] [LỖI] Quá trình đánh giá không sinh ra kết quả hợp lệ.")
        return 1

    print(f"\n[+] Kết quả mô hình [{detector.model_name}]:")
    print(f"    - Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"    - Precision: {metrics['precision']*100:.2f}%")
    print(f"    - Recall   : {metrics['recall']*100:.2f}%")
    print(f"    - F1-Score : {metrics['f1_score']*100:.2f}%")
    print(f"    - Tốc độ   : {metrics['fps']:.1f} FPS (Latency: {metrics['latency_ms']:.2f} ms)")
    print("\n" + metrics["report"])
    plot_confusion_matrix(metrics["confusion_matrix"], detector.model_name)
    return 0


def run_demo(model_name: str, input_path: str, is_scene: bool = False) -> int:
    """Chế độ Demo kiểm tra trên 1 ảnh đầu vào (ROI hoặc Scene toàn cảnh)."""
    path = Path(input_path)
    if not path.exists():
        print(f"[-] Không tìm thấy file ảnh: {path}")
        return 1

    img = imread_unicode(path)
    if img is None:
        print(f"[-] Không thể giải mã ảnh: {path}")
        return 1

    try:
        classifier = get_detector(model_name)
    except Exception as e:
        print(f"[-] [LỖI] {e}")
        return 1

    if not is_scene:
        print(f"\n🚀 Khởi chạy Demo với mô hình: [{classifier.model_name}] trên ảnh ROI biển báo...")
        try:
            res = classifier.predict(img)
        except Exception as e:
            print(f"[-] Lỗi suy luận: {e}")
            return 1

        print("=" * 50)
        print("🎯 KẾT QUẢ DỰ ĐOÁN:")
        print(f"  • Nhãn dự đoán   : {res.get('class_name', 'Unknown')} (Class ID: {res.get('class_id', -1)})")
        print(f"  • Độ tin cậy     : {res.get('confidence', 0.0) * 100:.2f}%")
        print(f"  • Thời gian xử lý: {res.get('inference_time', 0.0) * 1000:.2f} ms ({res.get('fps', 0.0):.1f} FPS)")
        print("=" * 50)
        return 0

    # Chế độ Scene: YOLO11s Detector + Classifier
    print(f"\n🚀 Khởi chạy Demo TOÀN CẢNH ĐƯỜNG PHỐ (Two-Stage: YOLO11s + {classifier.model_name})...")
    try:
        yolo_detector = load_yolo_detector()
    except Exception as e:
        print(f"[-] Không thể tải YOLO detector: {e}")
        return 1

    ih, iw = img.shape[:2]
    results = yolo_detector.predict(source=img, conf=0.25, iou=0.6, verbose=False, imgsz=800)

    boxes = []
    for r in results:
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, min(iw - 1, int(xyxy[0])))
                y1 = max(0, min(ih - 1, int(xyxy[1])))
                x2 = max(x1 + 2, min(iw, int(xyxy[2])))
                y2 = max(y1 + 2, min(ih, int(xyxy[3])))
                boxes.append((x1, y1, x2, y2))

    print(f"[+] YOLO11s đã phát hiện {len(boxes)} biển báo trên khung hình.")
    vis_img = img.copy()
    color_map = {
        "Bien_Cam": (0, 0, 255),        # Đỏ
        "Bien_Nguy_Hiem": (0, 215, 255), # Vàng
        "Bien_Hieu_Lenh": (255, 128, 0), # Xanh lam
        "Bien_Chi_Dan": (0, 200, 0),     # Xanh lá
        "Unknown": (128, 128, 128),
    }

    print("=" * 60)
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]
        pred = classifier.predict(crop)
        cls_name = pred.get("class_name", "Unknown")
        conf = pred.get("confidence", 0.0)
        col = color_map.get(cls_name, (200, 200, 200))

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), col, 2)
        label_text = f"{cls_name} ({conf*100:.1f}%)"
        cv2.putText(vis_img, label_text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        print(f"  Biển #{idx} [{x1},{y1},{x2},{y2}]: {label_text} | Phân loại: {pred.get('inference_time', 0)*1000:.2f} ms")
    print("=" * 60)

    out_demo = RESULTS_DIR / "demo_scene_result.jpg"
    ensure_dir(out_demo.parent)
    cv2.imwrite(str(out_demo), vis_img)
    print(f"[+] Đã lưu ảnh kết quả trực quan tại: {out_demo}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Hệ thống Benchmarking & Demo Nhận dạng Biển báo Giao thông")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["evaluate", "evaluate-scene", "demo"],
        help="Chế độ chạy: 'evaluate' (ROI Benchmark), 'evaluate-scene' (End-to-End Scene Benchmark), 'demo' (Thử nghiệm ảnh)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help="Chọn mô hình cụ thể ('color_hog_svm' hoặc 'deep_learning'). Để trống để benchmark tất cả.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Chọn tập dữ liệu để đánh giá: 'test' (mặc định) hoặc 'val'",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Số lượng mẫu tối đa khi đánh giá (mặc định: toàn bộ tập)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Đường dẫn file ảnh đầu vào (dùng trong chế độ demo)",
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        help="Bật chế độ thử nghiệm trên ảnh toàn cảnh đường phố (kết hợp YOLO11s)",
    )

    args = parser.parse_args()

    if args.samples is not None and args.samples <= 0:
        print(f"[-] [LỖI] Tham số --samples phải là số nguyên dương (> 0). Nhận được: {args.samples}")
        sys.exit(1)

    if args.mode == "evaluate":
        ret = run_evaluation(model_name=args.model, split_name=args.split, max_samples=args.samples)
        sys.exit(ret)
    elif args.mode == "evaluate-scene":
        try:
            run_scene_benchmark(split_name=args.split, max_samples=args.samples)
            sys.exit(0)
        except Exception as e:
            print(f"[-] [LỖI] Quá trình đánh giá toàn cảnh thất bại: {e}")
            sys.exit(1)
    elif args.mode == "demo":
        model_name = args.model or "deep_learning"
        if not args.input:
            print("[-] Vui lòng cung cấp đường dẫn ảnh bằng tham số --input khi chạy demo!")
            sys.exit(1)
        ret = run_demo(model_name=model_name, input_path=args.input, is_scene=args.scene)
        sys.exit(ret)


if __name__ == "__main__":
    main()
