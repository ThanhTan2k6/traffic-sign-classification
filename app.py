"""
Ứng dụng Giao diện Web Trực quan (Streamlit App) - Khung sườn cho Thành viên 3 phụ trách.
Chạy ứng dụng:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Nhận Diện Biển Báo Giao Thông",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Hệ Thống Phát Hiện & Nhận Dạng Biển Báo Giao Thông")
st.markdown("""
*Đồ án Môn học Thị giác máy tính - Quy trình 4 Giai đoạn: Tiền xử lý (HSV) ➔ Tìm ROI (Morphology + Contours) ➔ Trích xuất HOG ➔ Phân loại (SVM)*
""")

st.info("🚧 Khung sườn ứng dụng đang được Thành viên 3 phát triển và tích hợp.")
