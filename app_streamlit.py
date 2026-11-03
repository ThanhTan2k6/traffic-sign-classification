"""
================================================================================
🚦 GIAO DIỆN WEB DEMO TƯƠNG TÁC (STREAMLIT APP)
Đồ án Thị Giác Máy Tính: Nhận Dạng Biển Báo Giao Thông Việt Nam (QCVN 41:2019)
================================================================================
Hướng dẫn khởi chạy:
    streamlit run app_streamlit.py
================================================================================
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Đảm bảo đường dẫn gốc nhận diện đúng package
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TARGET_CLASSES, RESULTS_DIR
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Nhận Dạng Biển Báo Giao Thông (UIT)",
    page_icon="🚦",
    layout="wide",
)

VIETNAMESE_NAMES = {
    "Bien_Cam": "Biển Báo Cấm (P.*)",
    "Bien_Nguy_Hiem": "Biển Báo Nguy Hiểm & Cảnh Báo (W.*)",
    "Bien_Hieu_Lenh": "Biển Báo Hiệu Lệnh (R.*)",
    "Bien_Chi_Dan": "Biển Báo Chỉ Dẫn (I.*)",
    "Unknown": "Không xác định được biển báo",
}

COLOR_MAP = {
    "Bien_Cam": (231, 76, 60),        # Đỏ
    "Bien_Nguy_Hiem": (243, 156, 18),  # Vàng / Cam
    "Bien_Hieu_Lenh": (41, 128, 185),  # Xanh lam
    "Bien_Chi_Dan": (39, 174, 96),     # Xanh lá
    "Unknown": (127, 140, 141),        # Xám
}


@st.cache_resource
def load_detectors():
    """Nạp cache các mô hình một lần duy nhất."""
    svm_det = ColorHogSvmDetector()
    dl_det = DeepLearningDetector()
    return {"color_hog_svm": svm_det, "deep_learning": dl_det}


def main():
    st.title("🚦 Hệ Thống Nhận Dạng Biển Báo Giao Thông Việt Nam")
    st.markdown(
        "**Đồ án Thị Giác Máy Tính - ĐH Công Nghệ Thông Tin (UIT - VNU-HCM)**  \n"
        "So sánh & đối chuẩn giữa **Phương pháp Truyền thống (Color HOG + SVM)** và **Học Sâu (Enhanced LeNet-5)**."
    )

    detectors = load_detectors()

    # Thanh điều khiển bên trái
    st.sidebar.header("⚙️ Cấu Hình Thử Nghiệm")
    model_choice = st.sidebar.selectbox(
        "Chọn phương pháp / mô hình:",
        options=["deep_learning", "color_hog_svm"],
        format_func=lambda x: "Học Sâu: Enhanced LeNet-5 (Thành viên 2)" if x == "deep_learning" else "Truyền thống: Color HOG + SVM (Thành viên 1)",
    )

    detector = detectors[model_choice]

    tab1, tab2, tab3 = st.tabs(["🎯 Thử Nghiệm Ảnh Trực Quan", "📊 Bảng Xếp Hạng Benchmark", "📈 Phân Tích Dữ Liệu (EDA)"])

    # -------------------------------------------------------------
    # TAB 1: DEMO TRỰC QUAN
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Tải Lên Ảnh Để Nhận Dạng")
        col_input, col_result = st.columns([1, 1])

        with col_input:
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc chọn ảnh (JPG / PNG):",
                type=["jpg", "jpeg", "png"],
            )

            is_scene = st.checkbox(
                "Ảnh chụp toàn cảnh đường phố (Kích hoạt tự động định vị ROI)",
                value=False,
                help="Tích chọn nếu bạn tải ảnh chụp góc rộng đường phố. Bỏ tích nếu là ảnh biển báo đã cắt sẵn.",
            )

        if uploaded_file is not None:
            # Đọc ảnh từ buffer
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            with col_input:
                st.image(img_rgb, caption="Ảnh đầu vào", use_container_width=True)

            with col_result:
                st.subheader("Kết Quả Phân Loại")
                with st.spinner("Đang xử lý suy luận..."):
                    if is_scene and hasattr(detector, "predict_scene"):
                        res = detector.predict_scene(img_bgr)
                    elif is_scene and "is_roi" in detector.predict.__code__.co_varnames:
                        res = detector.predict(img_bgr, is_roi=False)
                    else:
                        res = detector.predict(img_bgr)

                pred_class = res.get("class_name", "Unknown")
                confidence = res.get("confidence", 0.0)
                fps = res.get("fps", 0.0)
                latency = res.get("inference_time", 0.0) * 1000.0
                bbox = res.get("bbox")

                st.success(f"**Nhóm Biển Báo:** {VIETNAMESE_NAMES.get(pred_class, pred_class)}")
                st.metric("Độ Tin Cậy", f"{confidence * 100:.2f}%")
                st.caption(f"⏱️ **Độ trễ:** {latency:.2f} ms | ⚡ **Tốc độ:** {fps:.1f} FPS")

                # Vẽ bounding box nếu có
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    annotated_rgb = img_rgb.copy()
                    color = COLOR_MAP.get(pred_class, (0, 255, 0))
                    cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        annotated_rgb,
                        f"{pred_class} ({confidence*100:.1f}%)",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                    st.image(annotated_rgb, caption="Vùng biển báo phát hiện được", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: BẢNG XẾP HẠNG BENCHMARK
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Bảng Xếp Hạng Kết Quả Đánh Giá Trên Tập Test Độc Lập (1.136 Mẫu)")

        leaderboard_data = {
            "Phương Pháp / Model": ["Enhanced LeNet-5 (Deep Learning)", "Color HOG + SVM (Truyền thống)"],
            "Người Phụ Trách": ["Thành viên 2", "Thành viên 1"],
            "Accuracy (%)": [99.74, 99.38],
            "Precision (%)": [99.49, 99.28],
            "Recall (%)": [99.69, 98.93],
            "Macro F1 (%)": [99.59, 99.10],
            "Độ Trễ (ms)": [1.12, 2.23],
            "Tốc Độ (FPS)": [890.6, 449.4],
        }
        df_lead = pd.DataFrame(leaderboard_data)
        st.dataframe(df_lead, use_container_width=True)

        benchmark_img = RESULTS_DIR / "benchmark_comparison.png"
        if benchmark_img.exists():
            st.image(str(benchmark_img), caption="Biểu đồ so sánh đa chiều giữa 2 mô hình", use_container_width=True)

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            cm_cnn = RESULTS_DIR / "confusion_matrix_deep_learning.png"
            if cm_cnn.exists():
                st.image(str(cm_cnn), caption="Confusion Matrix: Enhanced LeNet-5", use_container_width=True)
        with col_cm2:
            cm_svm = RESULTS_DIR / "confusion_matrix_color_hog_svm.png"
            if cm_svm.exists():
                st.image(str(cm_svm), caption="Confusion Matrix: Color HOG + SVM", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: PHÂN TÍCH DỮ LIỆU EDA
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Thống Kê Khám Phá Tập Dữ Liệu (EDA)")
        st.markdown(
            "Tập dữ liệu gốc gồm **3.216 ảnh chụp đường phố** tại Việt Nam với **8.334 bounding boxes**, "
            "được phân loại theo quy chuẩn **QCVN 41:2019**."
        )

        eda_dir = RESULTS_DIR / "eda"
        if eda_dir.exists():
            fig1 = eda_dir / "eda_1_target_distribution.png"
            fig2 = eda_dir / "eda_2_top20_classes.png"
            fig3 = eda_dir / "eda_3_box_sizes_aspect_ratio.png"
            fig4 = eda_dir / "eda_4_image_resolutions.png"

            if fig1.exists():
                st.image(str(fig1), caption="Phân bố 4 nhóm biển báo mục tiêu", use_container_width=True)
            if fig2.exists():
                st.image(str(fig2), caption="Top 20 biển báo xuất hiện phổ biến nhất", use_container_width=True)

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                if fig3.exists():
                    st.image(str(fig3), caption="Kích thước và tỉ lệ khung hình (Aspect Ratio)", use_container_width=True)
            with col_e2:
                if fig4.exists():
                    st.image(str(fig4), caption="Độ phân giải các ảnh chụp trong tập dữ liệu", use_container_width=True)


if __name__ == "__main__":
    main()
