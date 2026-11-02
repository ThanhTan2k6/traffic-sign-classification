"""
================================================================================
🚀 CONTROLLER CHÍNH DỰ ÁN (MAIN ENTRY POINT)
Hệ Thống Đánh Giá & So Sánh (Benchmarking) Các Phương Pháp Nhận Dạng Biển Báo
================================================================================
Hướng dẫn sử dụng:
1. Chạy đánh giá so sánh (Benchmark) tất cả các mô hình trên tập Test:
    python app.py --mode evaluate --split test

2. Chạy đánh giá 1 mô hình cụ thể:
    python app.py --mode evaluate --model color_hog_svm --split val

3. Chạy Demo thử nghiệm 1 mô hình trên ảnh bất kỳ:
    python app.py --mode demo --model color_hog_svm --input dataset/images/0001.jpg
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
    RESULTS_DIR,
)
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.evaluation.eval_metrics import evaluate_detector, benchmark_all_models
from src.evaluation.plotter import plot_benchmark_comparison, plot_confusion_matrix
from src.utils import imread_unicode


def get_detector(model_name: str):
    """Factory helper khởi tạo detector theo tên."""
    model_name = model_name.lower()
    if model_name == "color_hog_svm":
        return ColorHogSvmDetector()
    elif model_name == "deep_learning":
        return DeepLearningDetector()
    else:
        raise ValueError(f"Không hỗ trợ mô hình '{model_name}'. Hỗ trợ: {SUPPORTED_MODELS}")


def run_evaluation(model_name: str = None, split_name: str = "test"):
    """Chế độ Đánh giá & So sánh (Evaluation / Benchmark Mode)."""
    split_file = TEST_SPLIT_FILE if split_name.lower() == "test" else VAL_SPLIT_FILE

    if not split_file.exists():
        print(f"[-] [LỖI] Không tìm thấy file split: {split_file}")
        print("[!] Vui lòng chạy lệnh sau để khởi tạo dữ liệu trước:")
        print("    python global_data_split.py")
        return

    if model_name:
        # Đánh giá 1 model duy nhất
        detector = get_detector(model_name)
        metrics = evaluate_detector(detector, split_file)
        if metrics:
            print(f"\n[+] Kết quả mô hình [{model_name}]:")
            print(f"    - Accuracy : {metrics['accuracy']*100:.2f}%")
            print(f"    - F1-Score : {metrics['f1_score']*100:.2f}%")
            print(f"    - Tốc độ   : {metrics['fps']:.1f} FPS (Latency: {metrics['latency_ms']:.2f} ms)")
            print("\n" + metrics["report"])
            plot_confusion_matrix(metrics["confusion_matrix"], model_name)
    else:
        # Chạy Benchmarking tất cả các model (2 thành viên)
        detectors = [
            ColorHogSvmDetector(),
            DeepLearningDetector(),
        ]
        df_leaderboard = benchmark_all_models(detectors, split_file)
        if not df_leaderboard.empty:
            plot_benchmark_comparison(df_leaderboard)


def run_demo(model_name: str, input_path: str):
    """Chế độ Demo kiểm tra trên 1 ảnh đầu vào."""
    path = Path(input_path)
    if not path.exists():
        print(f"[-] Không tìm thấy file ảnh: {path}")
        return

    img = imread_unicode(path)
    if img is None:
        print(f"[-] Không thể giải mã ảnh: {path}")
        return

    detector = get_detector(model_name)
    print(f"\n🚀 Khởi chạy Demo với mô hình: [{model_name}]...")
    try:
        res = detector.predict(img)
    except NotImplementedError:
        print(f"[!] Mô hình [{model_name}] chưa được triển khai hoàn thiện (đang là skeleton).")
        print(f"    Thành viên phụ trách cần hoàn thành hàm preprocess, inference, postprocess trong thư mục src/detection/{model_name}/.")
        return
    except Exception as e:
        print(f"[-] Lỗi trong quá trình suy luận mô hình [{model_name}]: {e}")
        return

    print("=" * 50)
    print("🎯 KẾT QUẢ DỰ ĐOÁN:")
    print(f"  • Nhãn dự đoán   : {res.get('class_name', 'Unknown')} (Class ID: {res.get('class_id', -1)})")
    print(f"  • Độ tin cậy     : {res.get('confidence', 0.0) * 100:.2f}%")
    print(f"  • Thời gian xử lý: {res.get('inference_time', 0.0) * 1000:.2f} ms ({res.get('fps', 0.0):.1f} FPS)")
    if res.get("bbox") is not None:
        print(f"  • Bounding box   : {res.get('bbox')}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Hệ thống Benchmarking Nhận dạng Biển báo Giao thông")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["evaluate", "demo"],
        help="Chế độ chạy: 'evaluate' (đánh giá so sánh) hoặc 'demo' (chạy thử nghiệm ảnh)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help=f"Chọn mô hình cụ thể ({SUPPORTED_MODELS}). Nếu bỏ trống trong evaluate sẽ chạy so sánh toàn bộ.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Chọn tập dữ liệu để đánh giá: 'test' (mặc định) hoặc 'val'",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Đường dẫn file ảnh đầu vào (dùng trong chế độ demo)",
    )

    args = parser.parse_args()

    if args.mode == "evaluate":
        run_evaluation(model_name=args.model, split_name=args.split)
    elif args.mode == "demo":
        if not args.model:
            print("[-] Vui lòng chỉ định mô hình bằng tham số --model khi chạy demo!")
            return
        if not args.input:
            print("[-] Vui lòng chỉ định file ảnh đầu vào bằng tham số --input!")
            return
        run_demo(model_name=args.model, input_path=args.input)


if __name__ == "__main__":
    main()
