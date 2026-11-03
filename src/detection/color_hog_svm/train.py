"""
================================================================================
🚀 SCRIPT HUẤN LUYỆN: PHƯƠNG PHÁP 1 - HOG + COLOR HISTOGRAM + SVM (THÀNH VIÊN 1)
================================================================================
Quy trình:
1. Đọc dữ liệu từ Global Split (TRAIN_SPLIT_FILE, VAL_SPLIT_FILE).
2. Trích xuất đặc trưng Color + HOG (1796 chiều) từ ảnh ROI biển báo.
3. Chuẩn hóa đặc trưng (StandardScaler).
4. Huấn luyện mô hình Support Vector Machine (LinearSVC kết hợp CalibratedClassifierCV để sinh xác suất).
5. Đánh giá kiểm thử trên tập Validation và lưu checkpoint vào checkpoints/color_hog_svm/best_svm.pkl.
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, List
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

# Windows UTF-8 console output fix
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_TO_IDX,
    CROPPED_DATA_DIR,
    RANDOM_SEED,
    TARGET_CLASSES,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
)
from src.detection.color_hog_svm.features import extract_color_hog_features
from src.utils.image_io import imread_unicode


def load_dataset_features(split_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc file split và trích xuất vector đặc trưng cùng nhãn lớp."""
    if not split_file.exists():
        raise FileNotFoundError(f"[-] Không tìm thấy file split: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        lines = [l.strip().split("\t") for l in f if l.strip()]

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    desc = f"Trích xuất đặc trưng ({split_file.stem})"
    for parts in tqdm(lines, desc=desc):
        if len(parts) < 2:
            continue
        rel_path = parts[0]
        cat_name = parts[1]

        if cat_name not in CLASS_TO_IDX:
            continue

        img_path = CROPPED_DATA_DIR / rel_path
        img = imread_unicode(img_path)
        if img is None:
            continue

        feat = extract_color_hog_features(img)
        features_list.append(feat)
        labels_list.append(CLASS_TO_IDX[cat_name])

    if not features_list:
        raise RuntimeError(
            f"[-] Không trích xuất được mẫu nào từ {split_file}. "
            f"Kiểm tra ảnh ROI trong {CROPPED_DATA_DIR} (hãy chạy 'python global_data_split.py' để cắt ROI)."
        )

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def train_svm(
    c_param: float = 1.0,
    max_iter: int = 2000,
    seed: int = RANDOM_SEED,
) -> Path:
    print("=" * 70)
    print("🚀 HUẤN LUYỆN MÔ HÌNH HOG + COLOR HISTOGRAM + SVM (THÀNH VIÊN 1)")
    print("=" * 70)

    # 1. Trích xuất đặc trưng tập Train & Val
    print(f"[*] Nạp dữ liệu Train từ: {TRAIN_SPLIT_FILE}")
    X_train, y_train = load_dataset_features(TRAIN_SPLIT_FILE)
    print(f"    -> Đã trích xuất {len(X_train)} mẫu Train, kích thước vector: {X_train.shape[1]}")

    print(f"[*] Nạp dữ liệu Validation từ: {VAL_SPLIT_FILE}")
    X_val, y_val = load_dataset_features(VAL_SPLIT_FILE)
    print(f"    -> Đã trích xuất {len(X_val)} mẫu Val")

    # Đảm bảo toàn bộ 4 lớp mục tiêu đều có mặt trong tập huấn luyện
    unique_train_classes = set(y_train)
    if len(unique_train_classes) < len(TARGET_CLASSES):
        missing = [TARGET_CLASSES[i] for i in range(len(TARGET_CLASSES)) if i not in unique_train_classes]
        raise ValueError(f"[-] Tập dữ liệu huấn luyện không đủ 4 lớp, thiếu: {missing}")

    class_counts = [int(np.count_nonzero(y_train == class_idx)) for class_idx in range(len(TARGET_CLASSES))]
    min_samples = min(class_counts)
    if min_samples < 2:
        raise ValueError(f"[-] Mỗi lớp cần ít nhất 2 mẫu huấn luyện cho calibration, số mẫu nhỏ nhất hiện có: {min_samples}")
    calibration_folds = min(3, min_samples)

    # 2. Xây dựng Pipeline: StandardScaler + LinearSVC bên trong CalibratedClassifierCV
    # (Đảm bảo scaler được fit độc lập trong từng fold calibration, loại bỏ data leakage)
    print(f"\n[*] Khởi tạo mô hình Pipeline: StandardScaler + Calibrated LinearSVC (cv={calibration_folds})...")
    base_svc = LinearSVC(
        C=c_param,
        dual="auto",
        max_iter=max_iter,
        random_state=seed,
    )
    inner_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", base_svc),
    ])
    pipeline = CalibratedClassifierCV(inner_pipeline, cv=calibration_folds)

    # 3. Huấn luyện
    print(f"[*] Đang huấn luyện SVM (C={c_param}, max_iter={max_iter})...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - t0
    print(f"[+] Huấn luyện hoàn tất trong {train_duration:.2f} giây.")

    # 4. Đánh giá trên tập Train & Val
    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(
        y_val,
        val_preds,
        labels=list(range(len(TARGET_CLASSES))),
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HOG + SVM:")
    print(f"   • Train Accuracy : {train_acc * 100:.2f}%")
    print(f"   • Val Accuracy   : {val_acc * 100:.2f}%")
    print(f"   • Val Macro F1   : {val_f1 * 100:.2f}%")
    print("=" * 50)

    print("\n📋 Chi tiết báo cáo phân loại trên tập Validation:")
    print(classification_report(
        y_val,
        val_preds,
        labels=list(range(len(TARGET_CLASSES))),
        target_names=TARGET_CLASSES,
        digits=4,
        zero_division=0,
    ))

    # 5. Lưu mô hình checkpoint (Chỉ ghi đè nếu kết quả mới tốt hơn checkpoint hiện tại)
    save_dir = CHECKPOINTS_DIR / "color_hog_svm"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best_svm.pkl"

    should_save = True
    if save_path.exists():
        try:
            prev_data = joblib.load(save_path)
            prev_acc = prev_data.get("val_acc", 0.0) if isinstance(prev_data, dict) else 0.0
            if val_acc < prev_acc:
                should_save = False
                print(f"[*] Checkpoint hiện có (Val Acc: {prev_acc*100:.2f}%) tốt hơn kết quả hiện tại ({val_acc*100:.2f}%). Giữ nguyên checkpoint tốt nhất.")
        except Exception as e:
            print(f"[!] Không đọc được checkpoint cũ ({e}), tiến hành ghi mới.")

    if should_save:
        checkpoint_data = {
            "pipeline": pipeline,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "target_classes": TARGET_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
        }
        joblib.dump(checkpoint_data, save_path)
        print(f"\n[+] Đã lưu checkpoint tốt nhất tại: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Color HOG + SVM")
    parser.add_argument("--c", type=float, default=1.0, help="Tham số điều hòa C cho SVM (mặc định: 1.0)")
    parser.add_argument("--max-iter", type=int, default=2000, help="Số vòng lặp tối đa (mặc định: 2000)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Random seed (mặc định: {RANDOM_SEED})")

    args = parser.parse_args()
    train_svm(c_param=args.c, max_iter=args.max_iter, seed=args.seed)


if __name__ == "__main__":
    main()
