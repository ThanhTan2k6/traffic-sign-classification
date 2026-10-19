"""
Giai đoạn 4: Huấn luyện & Phân loại Biển báo (SVM / k-NN) - Thành viên 2 phụ trách.
Nhiệm vụ:
- Đọc dữ liệu ROI từ 'data/rois/'.
- Trích xuất đặc trưng HOG cho từng ảnh.
- Chia tập Train/Test theo tỷ lệ 80/20.
- Huấn luyện mô hình SVM (hỗ trợ tinh chỉnh C, gamma, kernel).
- Lưu mô hình và bộ mã hóa nhãn sang file .pkl để Thành viên 3 tích hợp.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from src.config import (
    ROI_OUTPUT_DIR,
    MODEL_DIR,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
)
from src.features.hog_extractor import HOGFeatureExtractor
from src.utils import ensure_dir, imread_unicode


def load_dataset_and_extract_hog(data_dir: Path):
    """
    Đọc tất cả ảnh ROI và trích xuất đặc trưng HOG.
    Trả về: X (mảng ma trận đặc trưng), y (mảng nhãn lớp).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Thư mục ROI chưa tồn tại: {data_path}. Vui lòng chạy bước tạo ROI của TV1 trước.")

    extractor = HOGFeatureExtractor()
    X = []
    y = []

    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"[+] Tìm thấy {len(class_dirs)} lớp trong {data_path}")

    supported_exts = {".png", ".jpg", ".jpeg", ".bmp"}

    for class_folder in class_dirs:
        class_name = class_folder.name
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in supported_exts]

        for img_file in tqdm(images, desc=f"HOG lớp {class_name:10s}", leave=False):
            img = imread_unicode(img_file)
            if img is None:
                continue
            features, _ = extractor.extract(img, visualize=False)
            X.append(features)
            y.append(class_name)

    return np.array(X), np.array(y)


def train_svm_pipeline(
    data_dir: Path = ROI_OUTPUT_DIR,
    model_save_path: Path = MODEL_PATH,
    encoder_save_path: Path = LABEL_ENCODER_PATH,
    use_knn: bool = False,
):
    """
    Quy trình huấn luyện hoàn chỉnh mô hình phân loại biển báo:
    """
    print("=" * 60)
    print("HUẤN LUYỆN MÔ HÌNH HỌC MÁY (SVM / k-NN)")
    print("=" * 60)

    # 1. Trích xuất đặc trưng
    X, y_labels = load_dataset_and_extract_hog(data_dir)
    print(f"\n[+] Kích thước ma trận đặc trưng X: {X.shape}")
    print(f"[+] Số lượng mẫu nhãn y: {len(y_labels)}")

    # 2. Mã hóa nhãn (Label Encoding)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    # 3. Phân chia Train/Test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[+] Số lượng mẫu Train (80%): {X_train.shape[0]}")
    print(f"[+] Số lượng mẫu Test (20%):  {X_test.shape[0]}")

    # 4. Khởi tạo và huấn luyện mô hình
    if use_knn:
        print("[+] Đang huấn luyện mô hình k-NN (k=5)...")
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    else:
        print(f"[+] Đang huấn luyện mô hình SVM (kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA})...")
        model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True, random_state=RANDOM_STATE)

    model.fit(X_train, y_train)

    # 5. Đánh giá sơ bộ trên tập Test
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Độ chính xác trên tập Test (Accuracy): {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 6. Lưu mô hình và encoder
    ensure_dir(model_save_path.parent)
    joblib.dump(model, model_save_path)
    joblib.dump(label_encoder, encoder_save_path)
    print(f"[+] Đã lưu trọng số mô hình tại: {model_save_path}")
    print(f"[+] Đã lưu Label Encoder tại:   {encoder_save_path}")

    return model, label_encoder


if __name__ == "__main__":
    train_svm_pipeline()
