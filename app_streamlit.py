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
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.evaluation.scene_eval import load_yolo_detector

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
def get_cached_lenet():
    """Cache mô hình Enhanced LeNet-5."""
    try:
        det = DeepLearningDetector()
        return det if det.model is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Lỗi tải DeepLearningDetector")
        return None


@st.cache_resource(max_entries=1)
def get_cached_svm():
    """Cache mô hình Color HOG SVM."""
    try:
        det = ColorHogSvmDetector()
        return det if det.model is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Lỗi tải ColorHogSvmDetector")
        return None


@st.cache_resource(max_entries=1)
def get_cached_yolo():
    """Cache mô hình định vị toàn cảnh YOLO11s."""
    try:
        return load_yolo_detector()
    except Exception:
        logging.getLogger(__name__).exception("Lỗi tải YOLO detector")
        return None


def main():
    st.title("🚦 Hệ Thống Nhận Dạng Biển Báo Giao Thông Việt Nam")
    st.markdown(
        "**Đồ án Thị Giác Máy Tính - ĐH Công Nghệ Thông Tin (UIT - VNU-HCM)**  \n"
        "Kiến trúc Đối chuẩn Đa Phương Pháp: **Cổ Điển (HSV + HOG + SVM - Thành viên 1)** đối đầu **Học Sâu (Enhanced LeNet-5 - Thành viên 2)**."
    )

    det_svm = get_cached_svm()
    det_lenet = get_cached_lenet()
    yolo_model = get_cached_yolo()

    # Thanh điều khiển bên trái
    st.sidebar.header("⚙️ Cấu Hình Thử Nghiệm")

    test_mode = st.sidebar.radio(
        "Chế độ thử nghiệm:",
        ["Ảnh Biển Báo Cắt Sẵn (ROI)", "Ảnh Toàn Cảnh Đường Phố (Scene)"],
        index=0,
    )

    chosen_model_name = st.sidebar.selectbox(
        "Mô hình phân loại:",
        [
            "Thành viên 2: Enhanced LeNet-5 (Học Sâu)",
            "Thành viên 1: Color + HOG + SVM (Truyền Thống)",
            "⚡ So sánh Song Song Cả 2 Mô Hình (Side-by-Side)",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Trạng Thái Checkpoints")
    st.sidebar.write("• **Enhanced LeNet-5:**", "✅ Sẵn sàng" if det_lenet is not None else "❌ Chưa tải")
    st.sidebar.write("• **Color HOG SVM:**", "✅ Sẵn sàng" if det_svm is not None else "❌ Chưa tải")
    st.sidebar.write("• **YOLO11s Detector:**", "✅ Sẵn sàng" if yolo_model is not None else "❌ Chưa tải")

    tab1, tab2, tab3 = st.tabs(["🎯 Thử Nghiệm Tương Tác", "🏆 Bảng Xếp Hạng Đối Chuẩn (2 Tầng)", "📈 Khám Phá Dữ Liệu (EDA)"])

    # -------------------------------------------------------------
    # TAB 1: DEMO TRỰC QUAN
    # -------------------------------------------------------------
    with tab1:
        if test_mode == "Ảnh Biển Báo Cắt Sẵn (ROI)":
            st.subheader("1. Thử Nghiệm Trên Ảnh Biển Báo Đã Cắt (ROI)")
            st.caption("Đưa vào một ảnh chụp cận cảnh biển báo giao thông để phân loại vào 4 nhóm QCVN 41:2019.")

            col_input, col_result = st.columns([1, 1])
            with col_input:
                uploaded_file = st.file_uploader(
                    "Chọn hoặc kéo thả ảnh biển báo (ROI - JPG / PNG):",
                    type=["jpg", "jpeg", "png"],
                    key="roi_uploader",
                )

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                with col_input:
                    st.image(img_rgb, caption="Ảnh biển báo đầu vào", use_container_width=True)

                with col_result:
                    st.subheader("🎯 Kết Quả Phân Loại")

                    if "Enhanced LeNet-5" in chosen_model_name:
                        if det_lenet is None:
                            st.error("Chưa nạp được checkpoint LeNet-5!")
                        else:
                            res = det_lenet.predict(img_bgr)
                            cls_name = res.get("class_name", "Unknown")
                            st.success(f"**Nhóm Biển:** {VIETNAMESE_NAMES.get(cls_name, cls_name)}")
                            st.metric("Độ Tin Cậy", f"{res.get('confidence', 0.0)*100:.2f}%")
                            st.caption(f"⏱️ **Độ trễ:** {res.get('inference_time', 0.0)*1000:.2f} ms | ⚡ **Tốc độ:** {res.get('fps', 0.0):.1f} FPS")

                    elif "Color + HOG + SVM" in chosen_model_name:
                        if det_svm is None:
                            st.error("Chưa nạp được checkpoint SVM!")
                        else:
                            res = det_svm.predict(img_bgr)
                            cls_name = res.get("class_name", "Unknown")
                            st.success(f"**Nhóm Biển:** {VIETNAMESE_NAMES.get(cls_name, cls_name)}")
                            st.metric("Độ Tin Cậy", f"{res.get('confidence', 0.0)*100:.2f}%")
                            st.caption(f"⏱️ **Độ trễ:** {res.get('inference_time', 0.0)*1000:.2f} ms | ⚡ **Tốc độ:** {res.get('fps', 0.0):.1f} FPS")

                    else:
                        # Side by Side
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.markdown("#### 👤 Thành viên 1: SVM")
                            if det_svm is not None:
                                r1 = det_svm.predict(img_bgr)
                                c1 = r1.get("class_name", "Unknown")
                                st.info(f"**{VIETNAMESE_NAMES.get(c1, c1)}**")
                                st.metric("Confidence", f"{r1.get('confidence', 0.0)*100:.1f}%")
                                st.caption(f"Latency: {r1.get('inference_time', 0)*1000:.2f} ms")
                            else:
                                st.warning("Chưa có model")

                        with col_m2:
                            st.markdown("#### 👤 Thành viên 2: LeNet-5")
                            if det_lenet is not None:
                                r2 = det_lenet.predict(img_bgr)
                                c2 = r2.get("class_name", "Unknown")
                                st.success(f"**{VIETNAMESE_NAMES.get(c2, c2)}**")
                                st.metric("Confidence", f"{r2.get('confidence', 0.0)*100:.1f}%")
                                st.caption(f"Latency: {r2.get('inference_time', 0)*1000:.2f} ms")
                            else:
                                st.warning("Chưa có model")

        else:
            # Scene Mode
            st.subheader("2. Thử Nghiệm Trên Ảnh Đường Phố Toàn Cảnh (Two-Stage Pipeline)")
            st.caption("Quy trình 2 tầng: YOLO11s tự động phát hiện vị trí biển báo -> Cắt ROI -> Phân loại.")

            uploaded_scene = st.file_uploader(
                "Chọn hoặc kéo thả ảnh đường phố toàn cảnh (JPG / PNG):",
                type=["jpg", "jpeg", "png"],
                key="scene_uploader",
            )

            if uploaded_scene is not None:
                file_bytes = np.asarray(bytearray(uploaded_scene.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                ih, iw = img_bgr.shape[:2]

                if yolo_model is None:
                    st.error("Không tìm thấy YOLO detector checkpoint tại `runs/detect/vn_s/weights/best.pt`!")
                else:
                    with st.spinner("YOLO11s đang quét và phân loại biển báo trên khung hình..."):
                        results = yolo_model.predict(source=img_bgr, conf=0.25, iou=0.6, verbose=False, imgsz=800)

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

                        vis_bgr = img_bgr.copy()
                        chosen_classifier = det_lenet if "LeNet-5" in chosen_model_name else det_svm
                        if chosen_classifier is None:
                            chosen_classifier = det_lenet or det_svm

                        detected_signs = []
                        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                            crop = img_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                pred = chosen_classifier.predict(crop)
                                cname = pred.get("class_name", "Unknown")
                                conf = pred.get("confidence", 0.0)
                                col_rgb = COLOR_MAP.get(cname, (200, 200, 200))
                                col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])

                                cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), col_bgr, 3)
                                label = f"#{idx} {cname} ({conf*100:.0f}%)"
                                cv2.putText(vis_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_bgr, 2)
                                detected_signs.append({
                                    "STT": idx,
                                    "Tọa độ [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                                    "Kích thước": f"{x2-x1}x{y2-y1} px",
                                    "Nhóm Dự Đoán": VIETNAMESE_NAMES.get(cname, cname),
                                    "Độ Tin Cậy": f"{conf*100:.1f}%",
                                    "Thời Gian": f"{pred.get('inference_time', 0)*1000:.2f} ms",
                                })

                    col_sc1, col_sc2 = st.columns([3, 2])
                    with col_sc1:
                        st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), caption=f"Phát hiện {len(boxes)} biển báo bằng YOLO11s", use_container_width=True)

                    with col_sc2:
                        st.markdown(f"### 📋 Chi Tiết Biển Báo Phát Hiện ({len(detected_signs)} biển)")
                        if detected_signs:
                            st.dataframe(pd.DataFrame(detected_signs), use_container_width=True)
                        else:
                            st.info("Không tìm thấy biển báo nào trên khung hình này với ngưỡng confidence >= 0.25.")

    # -------------------------------------------------------------
    # TAB 2: ĐỐI CHUẨN HIỆU NĂNG 2 TẦNG (BENCHMARK LEADERBOARD)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("🏆 Kết Quả Đối Chuẩn Thực Tế (2 Tầng Độc Lập)")

        st.markdown("#### 🥇 Tầng 1: Đánh Giá Phân Loại Trên Tập Blind Test (1.136 Mẫu ROI Cắt Sẵn)")
        st.caption("So sánh sòng phẳng năng lực phân loại độc lập của 2 mô hình trên tập test chuẩn quốc tế (không rò rỉ dữ liệu).")

        df_t1 = pd.DataFrame([
            {
                "Mô Hình / Phương Pháp": "Enhanced LeNet-5 (Học Sâu)",
                "Người Phụ Trách": "Thành viên 2",
                "Accuracy (%)": 99.74,
                "Precision (%)": 99.49,
                "Recall (%)": 99.69,
                "Macro F1 (%)": 99.59,
                "Độ Trễ (ms)": 1.30,
                "Tốc Độ (FPS)": 768.4,
            },
            {
                "Mô Hình / Phương Pháp": "Color + HOG + SVM (Cổ Điển)",
                "Người Phụ Trách": "Thành viên 1",
                "Accuracy (%)": 99.47,
                "Precision (%)": 98.14,
                "Recall (%)": 98.96,
                "Macro F1 (%)": 98.51,
                "Độ Trễ (ms)": 2.79,
                "Tốc Độ (FPS)": 357.9,
            },
        ])
        st.dataframe(df_t1, use_container_width=True)

        chart_roi = RESULTS_DIR / "benchmark_comparison.png"
        if chart_roi.exists():
            st.image(str(chart_roi), caption="Biểu đồ đối chuẩn Năng lực phân loại (Accuracy, F1, Latency, FPS)", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🥈 Tầng 2: Đánh Giá Toàn Trình Trên Ảnh Đường Phố Toàn Cảnh (End-to-End Scene Benchmark)")
        st.caption("Đánh giá hệ thống 2 giai đoạn: YOLO11s Detector + Bộ phân loại thứ cấp trên ảnh đường phố độ phân giải cao.")

        df_t2 = pd.DataFrame([
            {
                "Pipeline Toàn Trình": "Two-Stage LeNet (YOLO11s + LeNet-5)",
                "Tỷ Lệ Định Vị (%)": "85.61%",
                "Đoán Đúng Khi Tìm Thấy (%)": "99.40%",
                "E2E Recall (%)": "85.10%",
                "Độ Trễ Toàn Cảnh": "91.45 ms",
                "Tốc Độ (FPS)": "10.9 FPS",
            },
            {
                "Pipeline Toàn Trình": "Two-Stage SVM (YOLO11s + SVM RBF)",
                "Tỷ Lệ Định Vị (%)": "85.61%",
                "Đoán Đúng Khi Tìm Thấy (%)": "99.17%",
                "E2E Recall (%)": "84.90%",
                "Độ Trễ Toàn Cảnh": "105.20 ms",
                "Tốc Độ (FPS)": "9.5 FPS",
            },
        ])
        st.dataframe(df_t2, use_container_width=True)

        chart_scene = RESULTS_DIR / "scene_benchmark_comparison.png"
        if chart_scene.exists():
            st.image(str(chart_scene), caption="Biểu đồ đối chuẩn Toàn cảnh đường phố (End-to-End)", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: PHÂN TÍCH DỮ LIỆU EDA
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Thống Kê Khám Phá Tập Dữ Liệu (EDA)")
        st.markdown(
            "Tập dữ liệu gồm **3.216 ảnh chụp đường phố** tại Việt Nam với **8.334 bounding boxes**, "
            "được gom thành 4 nhóm chính theo quy chuẩn **QCVN 41:2019**."
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
