"""
================================================================================
🚀 CONTROLLER CHÍNH DỰ ÁN (MAIN ENTRY POINT)
Hệ Thống Đánh Giá & Thử Nghiệm Mô Hình Nhận Dạng Biển Báo Giao Thông
================================================================================
Hướng dẫn sử dụng:
1. Chạy đánh giá (Benchmark) mô hình trên tập Test:
    python app.py --mode evaluate --split test

2. Chạy đánh giá trên tập Val với giới hạn mẫu:
    python app.py --mode evaluate --model deep_learning --split val --samples 50

3. Chạy Demo thử nghiệm mô hình trên ảnh ROI biển báo:
    python app.py --mode demo --model deep_learning --input data/cropped/test/Bien_Cam/0002_00.jpg
================================================================================
"""

import argparse
import sys
from pathlib import Path

# UTF-8 stdout/stderr fix cho Windows
if sys.platform == "win32":
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
)
from src.detection.deep_learning import DeepLearningDetector
from src.evaluation.eval_metrics import evaluate_detector, benchmark_all_models
from src.evaluation.plotter import plot_benchmark_comparison, plot_confusion_matrix
from src.utils import imread_unicode


def get_detector(model_name: str = "deep_learning"):
    """Factory helper khởi tạo detector theo tên."""
    model_name = (model_name or "deep_learning").lower()
    if model_name != "deep_learning":
        raise ValueError(
            f"Mô hình '{model_name}' không được hỗ trợ trong phiên bản của Thành viên 2. "
            f"Chỉ hỗ trợ: ['deep_learning']"
        )
    cnn_ckpt = CHECKPOINTS_DIR / "deep_learning" / "best_cnn.pth"

    if not cnn_ckpt.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint mô hình tại: {cnn_ckpt}. "
            "Vui lòng huấn luyện mô hình trước: python -m src.detection.deep_learning.train_cnn"
        )
    detector = DeepLearningDetector(checkpoint_path=str(cnn_ckpt))
    if detector.model is None:
        raise RuntimeError(
            f"Không thể nạp trọng số mô hình từ checkpoint: {cnn_ckpt}"
        )
    return detector


def run_evaluation(model_name: str = None, split_name: str = "test", max_samples: int = None) -> int:
    """Chế độ Đánh giá & So sánh (Evaluation / Benchmark Mode)."""
    split_file = TEST_SPLIT_FILE if split_name.lower() == "test" else VAL_SPLIT_FILE

    if not split_file.exists():
        print(f"[-] [LỖI] Không tìm thấy file split: {split_file}")
        return 1

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
    print(f"    - F1-Score : {metrics['f1_score']*100:.2f}%")
    print(f"    - Tốc độ   : {metrics['fps']:.1f} FPS (Latency: {metrics['latency_ms']:.2f} ms)")
    print("\n" + metrics["report"])
    plot_confusion_matrix(metrics["confusion_matrix"], detector.model_name)
    return 0


def run_demo(model_name: str, input_path: str) -> int:
    """Chế độ Demo kiểm tra trên 1 ảnh đầu vào ROI biển báo."""
    path = Path(input_path)
    if not path.exists():
        print(f"[-] Không tìm thấy file ảnh: {path}")
        return 1

    img = imread_unicode(path)
    if img is None:
        print(f"[-] Không thể giải mã ảnh: {path}")
        return 1

    try:
        detector = get_detector(model_name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"[-] [LỖI] {e}")
        return 1

    print(f"\n🚀 Khởi chạy Demo với mô hình: [{detector.model_name}] trên ảnh ROI biển báo...")
    try:
        res = detector.predict(img)
    except NotImplementedError:
        print(f"[!] Mô hình [{detector.model_name}] chưa nạp checkpoint!")
        return 1
    except Exception as e:
        print(f"[-] Lỗi trong quá trình suy luận mô hình [{detector.model_name}]: {e}")
        return 1

    print("=" * 50)
    print("🎯 KẾT QUẢ DỰ ĐOÁN:")
    print(f"  • Nhãn dự đoán   : {res.get('class_name', 'Unknown')} (Class ID: {res.get('class_id', -1)})")
    print(f"  • Độ tin cậy     : {res.get('confidence', 0.0) * 100:.2f}%")
    print(f"  • Thời gian xử lý: {res.get('inference_time', 0.0) * 1000:.2f} ms ({res.get('fps', 0.0):.1f} FPS)")
    print("=" * 50)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Hệ thống Benchmarking Nhận dạng Biển báo Giao thông")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["evaluate", "demo"],
        help="Chế độ chạy: 'evaluate' (đánh giá trên tập test) hoặc 'demo' (chạy thử nghiệm ảnh ROI)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deep_learning",
        choices=["deep_learning"],
        help="Chọn mô hình cụ thể (chỉ hỗ trợ 'deep_learning').",
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

    args = parser.parse_args()

    if args.samples is not None and args.samples <= 0:
        print(f"[-] [LỖI] Tham số --samples phải là số nguyên dương (> 0). Nhận được: {args.samples}")
        sys.exit(1)

    if args.mode == "evaluate":
        ret = run_evaluation(model_name=args.model, split_name=args.split, max_samples=args.samples)
        if ret != 0:
            sys.exit(ret)
    elif args.mode == "demo":
        model_name = args.model or "deep_learning"
        if not args.input:
            print("[-] Vui lòng cung cấp đường dẫn ảnh bằng tham số --input khi chạy demo!")
            sys.exit(1)
        ret = run_demo(model_name=model_name, input_path=args.input)
        if ret != 0:
            sys.exit(ret)


if __name__ == "__main__":
    main()
