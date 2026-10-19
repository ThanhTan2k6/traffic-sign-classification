"""
Module Đánh giá Mô hình (Model Evaluation) - Thành viên 2 & 3.
Nhiệm vụ:
- Tính toán Accuracy, Precision, Recall, F1-Score.
- Trực quan hóa ma trận nhầm lẫn (Confusion Matrix).
- Xuất biểu đồ phục vụ Slide và Báo cáo đồ án.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import MODEL_PATH, LABEL_ENCODER_PATH, ROI_OUTPUT_DIR
from src.models.train_svm import load_dataset_and_extract_hog


def evaluate_model(
    model_path: Path = MODEL_PATH,
    encoder_path: Path = LABEL_ENCODER_PATH,
    data_dir: Path = ROI_OUTPUT_DIR,
    save_fig_path: Path = None,
):
    """Đánh giá chi tiết mô hình đã huấn luyện và vẽ ma trận nhầm lẫn."""
    if not model_path.exists() or not encoder_path.exists():
        print("[-] Chưa tìm thấy file mô hình hoặc nhãn encoder!")
        return

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    X, y_labels = load_dataset_and_extract_hog(data_dir)
    y_true = label_encoder.transform(y_labels)
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[+] Tổng thể Accuracy: {acc * 100:.2f}%")
    print("\n--- Báo cáo Phân loại Chi tiết (Classification Report) ---")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Ma trận nhầm lẫn (Confusion Matrix)", fontsize=14, fontweight="bold")
    plt.xlabel("Nhãn Dự đoán (Predicted)", fontsize=12)
    plt.ylabel("Nhãn Thực tế (Actual)", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
        print(f"[+] Đã lưu biểu đồ Confusion Matrix vào: {save_fig_path}")

    plt.show()


if __name__ == "__main__":
    evaluate_model()
