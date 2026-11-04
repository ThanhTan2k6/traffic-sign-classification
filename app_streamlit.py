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
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Đảm bảo đường dẫn gốc nhận diện đúng package
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TARGET_CLASSES, RESULTS_DIR, CHECKPOINTS_DIR
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


@st.cache_resource(max_entries=1)
def _get_cached_detector(ckpt_mtime: float, ckpt_path: str):
    """Nạp và cache mô hình dựa trên mtime của checkpoint. Không cache kết quả tải thất bại."""
    detector = DeepLearningDetector(checkpoint_path=ckpt_path)
    if detector.model is None:
        raise RuntimeError(f"Không thể nạp trọng số mô hình từ: {ckpt_path}")
    return detector


def load_detector():
    """Kiểm tra checkpoint bên ngoài hàm cache để tránh cache vĩnh viễn trạng thái None khi chưa có file."""
    cnn_ckpt = CHECKPOINTS_DIR / "deep_learning" / "best_cnn.pth"
    if not cnn_ckpt.exists():
        return None
    try:
        return _get_cached_detector(cnn_ckpt.stat().st_mtime, str(cnn_ckpt))
    except Exception:
        logging.getLogger(__name__).exception("Không thể nạp detector từ %s", cnn_ckpt)
        return None


def main():
    st.title("🚦 Hệ Thống Nhận Dạng Biển Báo Giao Thông Việt Nam")
    st.markdown(
        "**Đồ án Thị Giác Máy Tính - ĐH Công Nghệ Thông Tin (UIT - VNU-HCM)**  \n"
        "Phương pháp Học Sâu: **Enhanced LeNet-5 (Thành viên 2)**."
    )

    detector = load_detector()

    # Thanh điều khiển bên trái
    st.sidebar.header("⚙️ Cấu Hình Thử Nghiệm")
    if detector is not None and detector.model is not None:
        model_label = "Enhanced LeNet-5 (PyTorch CNN)" if detector.arch_type == "cnn" else "Deep Learning Classifier"
        st.sidebar.info(f"Mô hình hoạt động: **{model_label}**")
    else:
        st.sidebar.warning("Mô hình: **Chưa nạp checkpoint (Cần huấn luyện)**")

    tab1, tab2, tab3 = st.tabs(["🎯 Thử Nghiệm Ảnh Trực Quan", "📊 Bảng Xếp Hạng Benchmark", "📈 Phân Tích Dữ Liệu (EDA)"])

    # -------------------------------------------------------------
    # TAB 1: DEMO TRỰC QUAN
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Tải Lên Ảnh Biển Báo Để Nhận Dạng (ROI)")
        st.caption("Chế độ phân loại ảnh biển báo đã cắt theo chuẩn bài toán Traffic Sign Classification.")
        col_input, col_result = st.columns([1, 1])

        with col_input:
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc chọn ảnh biển báo (ROI - JPG / PNG):",
                type=["jpg", "jpeg", "png"],
            )

        if uploaded_file is not None:
            # Đọc ảnh từ buffer
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            with col_input:
                st.image(img_rgb, caption="Ảnh biển báo đầu vào", use_container_width=True)

            with col_result:
                st.subheader("Kết Quả Phân Loại")
                cnn_ckpt = CHECKPOINTS_DIR / "deep_learning" / "best_cnn.pth"
                if not cnn_ckpt.exists():
                    st.error(
                        f"[-] Không tìm thấy checkpoint mô hình Enhanced LeNet-5 tại: `{cnn_ckpt}`. "
                        "Vui lòng chạy script huấn luyện: `python -m src.detection.deep_learning.train_cnn`"
                    )
                elif detector is None or detector.model is None:
                    st.error(
                        f"[-] Không thể nạp trọng số từ checkpoint `{cnn_ckpt}`. "
                        "Vui lòng kiểm tra tính tương thích của file checkpoint hoặc log server."
                    )
                else:
                    with st.spinner("Đang xử lý suy luận..."):
                        res = detector.predict(img_bgr)

                    pred_class = res.get("class_name", "Unknown")
                    confidence = res.get("confidence", 0.0)
                    fps = res.get("fps", 0.0)
                    latency = res.get("inference_time", 0.0) * 1000.0

                    st.success(f"**Nhóm Biển Báo:** {VIETNAMESE_NAMES.get(pred_class, pred_class)}")
                    st.metric("Độ Tin Cậy", f"{confidence * 100:.2f}%")
                    st.caption(f"⏱️ **Độ trễ:** {latency:.2f} ms | ⚡ **Tốc độ:** {fps:.1f} FPS")

    # -------------------------------------------------------------
    # TAB 2: BẢNG XẾP HẠNG BENCHMARK (ENHANCED LENET-5)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("🥇 Đánh Giá Phân Loại Trên Tập Blind Test (ROI - 1.136 Mẫu)")
        st.caption("Đánh giá mô hình học sâu Enhanced LeNet-5 trên tập Blind Test cắt sẵn theo chuẩn quốc tế.")

        leaderboard_data = {
            "Phương Pháp / Model": ["Enhanced LeNet-5 (Deep Learning)"],
            "Người Phụ Trách": ["Thành viên 2"],
            "Accuracy (%)": [99.74],
            "Precision (%)": [99.49],
            "Recall (%)": [99.69],
            "Macro F1 (%)": [99.59],
            "Độ Trễ (ms)": [1.12],
            "Tốc Độ (FPS)": [890.6],
        }
        df_lead = pd.DataFrame(leaderboard_data)
        st.dataframe(df_lead, use_container_width=True)

        cm_cnn = RESULTS_DIR / "confusion_matrix_deep_learning.png"
        if cm_cnn.exists():
            st.image(str(cm_cnn), caption="Confusion Matrix: Enhanced LeNet-5 (Deep Learning)", use_container_width=True)

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
