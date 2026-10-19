"""
Ứng dụng Giao diện Web Trực quan (Streamlit App) - Thành viên 3 phụ trách.
Hỗ trợ:
- Tải ảnh lên từ máy hoặc chọn ảnh mẫu trong dataset.
- Trực quan hóa từng bước trong luồng 4 giai đoạn:
  + Giai đoạn 1: Ảnh sau lọc màu HSV (Đỏ, Xanh, Vàng).
  + Giai đoạn 2: Khử nhiễu hình thái học, Contours, Cắt vùng ROI.
  + Giai đoạn 3: Biểu diễn trực quan đặc trưng HOG.
  + Giai đoạn 4: Dự đoán nhãn lớp và hiển thị độ tin cậy.
- Hỗ trợ chế độ nhận diện thời gian thực qua Camera/Webcam.

Chạy ứng dụng:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.detection.color_filter import ColorFilter
from src.detection.roi_extractor import ROIExtractor
from src.features.hog_extractor import HOGFeatureExtractor
from src.pipeline.end_to_end import TrafficSignPipeline

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Nhận Diện Biển Báo Giao Thông",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Hệ Thống Phát Hiện & Nhận Dạng Biển Báo Giao Thông")
st.markdown("""
*Đồ án Môn học Thị giác máy tính - Quy trình 4 Giai đoạn: Tiền xử lý (HSV) ➔ Tìm ROI (Morphology + Contours) ➔ Trích xuất HOG ➔ Phân loại (SVM)*
""")

# Khởi tạo Pipeline
@st.cache_resource
def load_pipeline():
    return TrafficSignPipeline()

pipeline = load_pipeline()
color_filter = ColorFilter()
roi_extractor = ROIExtractor()
hog_extractor = HOGFeatureExtractor()

# Sidebar cấu hình
st.sidebar.header("⚙️ Cấu hình phát hiện")
source_mode = st.sidebar.radio("Nguồn ảnh đầu vào:", ["Tải ảnh lên (Upload)", "Camera / Webcam"])

min_area = st.sidebar.slider("Diện tích tối thiểu (Min Area):", 50, 2000, 100, step=50)
roi_extractor.min_area = min_area

uploaded_file = None
if source_mode == "Tải ảnh lên (Upload)":
    uploaded_file = st.sidebar.file_uploader("Chọn ảnh cần nhận dạng...", type=["jpg", "jpeg", "png", "bmp"])
else:
    uploaded_file = st.sidebar.camera_input("Chụp ảnh từ Webcam")

if uploaded_file is not None:
    # Đọc ảnh sang numpy array BGR
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Thực thi Pipeline
    result = pipeline.predict(bgr_img)
    annotated = result["annotated_image"]
    mask = result["mask"]
    masked_img = result["masked_image"]
    predictions = result["predictions"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Ảnh gốc đầu vào")
        st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("🎯 Kết quả Nhận diện & Khoanh vùng")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Hiển thị các bước trung gian (Visualized Stages)
    st.markdown("---")
    st.subheader("🔬 Phân tích chi tiết các giai đoạn xử lý")

    tab1, tab2, tab3 = st.tabs(["1. Lọc màu HSV (Color Mask)", "2. Cắt vùng ROI (Bounding Boxes)", "3. Đặc trưng HOG (Features)"])

    with tab1:
        st.write("**Mặt nạ nhị phân và ảnh sau khi áp bộ lọc màu (Đỏ, Xanh, Vàng):**")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.image(mask, caption="Mặt nạ nhị phân (Binary Mask)", use_container_width=True)
        with m_col2:
            st.image(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), caption="Ảnh áp mặt nạ (Color Filtered)", use_container_width=True)

    with tab2:
        st.write(f"**Số lượng biển báo phát hiện: {len(predictions)}**")
        if predictions:
            roi_cols = st.columns(min(len(predictions), 4))
            for i, p in enumerate(predictions):
                with roi_cols[i % 4]:
                    st.image(
                        cv2.cvtColor(p["roi_normalized"], cv2.COLOR_BGR2RGB),
                        caption=f"ROI #{i+1}: {p['label']}\nConf: {p['confidence']*100:.1f}%",
                        width=120,
                    )
        else:
            st.info("Không phát hiện vùng ROI nào thỏa mãn điều kiện lọc.")

    with tab3:
        if predictions:
            sample_roi = predictions[0]["roi_normalized"]
            features, hog_vis = hog_extractor.extract(sample_roi, visualize=True)
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.image(cv2.cvtColor(sample_roi, cv2.COLOR_BGR2RGB), caption="ROI chuẩn hóa 64x64", width=150)
            with h_col2:
                st.image(hog_vis, caption=f"Trực quan hóa HOG (Vector {len(features)} chiều)", width=150)
        else:
            st.info("Chưa có ROI nào để trích xuất HOG.")
else:
    st.info("👈 Vui lòng tải một ảnh lên hoặc chụp từ webcam qua thanh điều khiển bên trái để bắt đầu.")
