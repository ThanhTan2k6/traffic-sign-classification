"""
================================================================================
🚦 GIAO DIỆN WEB DEMO TƯƠNG TÁC HIỆN ĐẠI (STREAMLIT DASHBOARD)
Đồ án Thị Giác Máy Tính (CS231) - ĐH Công Nghệ Thông Tin (UIT - VNU-HCM)
Hệ Thống Nhận Dạng Biển Báo Giao Thông Việt Nam (QCVN 41:2019)
Kiến trúc Đối chuẩn: Cổ Điển (Color+HOG+SVM) vs Học Sâu (Enhanced LeNet-5)
================================================================================
Lệnh khởi chạy (Windows / Powershell):
    python -m streamlit run app_streamlit.py
================================================================================
"""

import sys
import logging
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Import Plotly cho biểu đồ tương tác đẹp mắt
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Đảm bảo đường dẫn gốc nhận diện đúng package
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TARGET_CLASSES, RESULTS_DIR, CHECKPOINTS_DIR, DATASET_DIR
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.evaluation.scene_eval import load_yolo_detector

# ==============================================================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Hệ Thống Nhận Dạng Biển Báo Giao Thông (UIT - CS231)",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# BẢNG THÔNG TIN VÀ MÀU SẮC CHUẨN QCVN 41:2019
# ==============================================================================
CLASS_METADATA = {
    "Bien_Cam": {
        "vn_name": "Biển Báo Cấm",
        "code_prefix": "P.*",
        "color_hex": "#EF4444",
        "color_rgb": (239, 68, 68),
        "color_bgr": (68, 68, 239),
        "shape": "Hình tròn, viền đỏ, nền trắng, hình vẽ màu đen",
        "badge_emoji": "🚫",
        "description": "Báo hiệu các điều cấm mà người tham gia giao thông không được vi phạm.",
    },
    "Bien_Nguy_Hiem": {
        "vn_name": "Biển Nguy Hiểm & Cảnh Báo",
        "code_prefix": "W.*",
        "color_hex": "#F59E0B",
        "color_rgb": (245, 158, 11),
        "color_bgr": (11, 158, 245),
        "shape": "Hình tam giác đều, viền đỏ, nền vàng, hình vẽ màu đen",
        "badge_emoji": "⚠️",
        "description": "Cảnh báo trước các tình huống nguy hiểm có thể xảy ra trên đoạn đường.",
    },
    "Bien_Hieu_Lenh": {
        "vn_name": "Biển Báo Hiệu Lệnh",
        "code_prefix": "R.*",
        "color_hex": "#3B82F6",
        "color_rgb": (59, 130, 246),
        "color_bgr": (246, 130, 59),
        "shape": "Hình tròn, nền xanh lam, hình vẽ màu trắng",
        "badge_emoji": "🔵",
        "description": "Báo hiệu các hiệu lệnh người tham gia giao thông bắt buộc phải thi hành.",
    },
    "Bien_Chi_Dan": {
        "vn_name": "Biển Báo Chỉ Dẫn",
        "code_prefix": "I.*",
        "color_hex": "#10B981",
        "color_rgb": (16, 185, 129),
        "color_bgr": (129, 185, 16),
        "shape": "Hình chữ nhật hoặc hình vuông, nền xanh lam/xanh lá, hình vẽ trắng",
        "badge_emoji": "ℹ️",
        "description": "Hướng dẫn thông tin cần thiết về đường sá, địa danh, tiện ích giao thông.",
    },
    "Unknown": {
        "vn_name": "Không Xác Định",
        "code_prefix": "?",
        "color_hex": "#9CA3AF",
        "color_rgb": (156, 163, 175),
        "color_bgr": (175, 163, 156),
        "shape": "Không xác định",
        "badge_emoji": "❓",
        "description": "Mô hình không đủ độ tin cậy để kết luận nhóm biển báo.",
    },
}

PRESET_SAMPLES = {
    "cam": {
        "label": "Biển Cấm (P.102 - Cấm đi ngược chiều)",
        "icon": "🚫",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Cam" / "0306_00.jpg",
        "true_class": "Bien_Cam",
    },
    "nguy_hiem": {
        "label": "Biển Nguy Hiểm (W.221b - Đường gồ ghề)",
        "icon": "⚠️",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Nguy_Hiem" / "0357_00.jpg",
        "true_class": "Bien_Nguy_Hiem",
    },
    "hieu_lenh": {
        "label": "Biển Hiệu Lệnh (R.302a - Hướng phải đi)",
        "icon": "🔵",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Hieu_Lenh" / "0357_01.jpg",
        "true_class": "Bien_Hieu_Lenh",
    },
    "chi_dan": {
        "label": "Biển Chỉ Dẫn (I.434a - Bến xe buýt)",
        "icon": "ℹ️",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Chi_Dan" / "0307_01.jpg",
        "true_class": "Bien_Chi_Dan",
    },
}

PRESET_SCENES = {
    "scene_1": {
        "label": "Toàn cảnh đô thị 01 (0001.jpg)",
        "path": PROJECT_ROOT / "dataset" / "images" / "0001.jpg",
    },
    "scene_2": {
        "label": "Toàn cảnh giao lộ 02 (0002.jpg)",
        "path": PROJECT_ROOT / "dataset" / "images" / "0002.jpg",
    },
}

# ==============================================================================
# CSS STYLING: MODERN GLASSMORPHISM & TECH DASHBOARD
# ==============================================================================
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(14, 30, 64, 0.95) 0%, rgba(30, 58, 138, 0.90) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #93C5FD, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #CBD5E1;
        font-size: 14.5px;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-tag {
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .hero-tag-green {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .hero-tag-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FCD34D;
        border-color: rgba(245, 158, 11, 0.3);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease-in-out;
    }
    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }

    /* Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin: 12px 0;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 11.5px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Class Result Banner */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 12px;
    }
    .badge-cam {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1.5px solid #EF4444;
    }
    .badge-nguy-hiem {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1.5px solid #F59E0B;
    }
    .badge-hieu-lenh {
        background: rgba(59, 130, 246, 0.18);
        color: #60A5FA;
        border: 1.5px solid #3B82F6;
    }
    .badge-chi-dan {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1.5px solid #10B981;
    }
    .badge-unknown {
        background: rgba(156, 163, 175, 0.18);
        color: #D1D5DB;
        border: 1.5px solid #9CA3AF;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-size: 14px;
    }
</style>
"""


# ==============================================================================
# HÀM TẢI VÀ CACHE MÔ HÌNH
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_lenet():
    """Cache mô hình Enhanced LeNet-5."""
    try:
        det = DeepLearningDetector()
        return det if det.model is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Lỗi nạp Enhanced LeNet-5")
        return None


@st.cache_resource(show_spinner=False)
def get_cached_svm():
    """Cache mô hình Color + HOG + SVM."""
    try:
        det = ColorHogSvmDetector()
        return det if det.model is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Lỗi nạp ColorHogSVM")
        return None


@st.cache_resource(show_spinner=False)
def get_cached_yolo():
    """Cache mô hình YOLO11s phát hiện toàn cảnh."""
    try:
        return load_yolo_detector()
    except Exception:
        logging.getLogger(__name__).exception("Lỗi nạp YOLO11s")
        return None


def extract_probabilities(detector, img_bgr, model_type="lenet"):
    """
    Trích xuất phân phối xác suất trên 4 nhóm biển báo chuẩn.
    Trả về dict: {class_name: float_prob}
    """
    default_probs = {cls: 0.0 for cls in TARGET_CLASSES}
    if detector is None or img_bgr is None:
        return default_probs

    try:
        if model_type == "lenet":
            import torch
            prep = detector.preprocess(img_bgr)
            logits = detector.inference(prep)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            from src.config import IDX_TO_CLASS
            return {IDX_TO_CLASS.get(i, f"Class_{i}"): float(probs[i]) for i in range(len(probs))}

        elif model_type == "svm":
            from scipy.special import softmax
            prep = detector.preprocess(img_bgr)
            if hasattr(detector.model, "decision_function"):
                df = detector.model.decision_function(prep)
                raw_probs = softmax(df, axis=1)[0]
                classes = getattr(detector, "classes_", None)
                if classes is None and hasattr(detector.model, "classes_"):
                    classes = detector.model.classes_

                prob_dict = {cls: 0.0 for cls in TARGET_CLASSES}
                if classes is not None:
                    for raw_label, prob_val in zip(classes, raw_probs):
                        mapped = detector.LABEL_MAPPING.get(str(raw_label), str(raw_label))
                        if mapped in prob_dict:
                            prob_dict[mapped] = float(prob_val)
                    # Chuẩn hóa về tổng = 1 trên 4 class
                    total = sum(prob_dict.values())
                    if total > 0:
                        for k in prob_dict:
                            prob_dict[k] /= total
                    return prob_dict
    except Exception as e:
        logging.getLogger(__name__).debug(f"Trích xuất probabilities lỗi: {e}")

    return default_probs


def plot_probability_bar_chart(prob_dict, title="Phân Phối Độ Tin Cậy Trên 4 Nhóm Biển Báo"):
    """Vẽ biểu đồ thanh phân phối xác suất bằng Plotly với màu chuẩn QCVN."""
    if not HAS_PLOTLY:
        # Fallback Streamlit native bar chart
        df = pd.DataFrame(list(prob_dict.items()), columns=["Nhóm Biển", "Độ Tin Cậy"])
        st.bar_chart(df.set_index("Nhóm Biển"))
        return

    labels = []
    values = []
    colors = []
    for cls in TARGET_CLASSES:
        meta = CLASS_METADATA.get(cls, CLASS_METADATA["Unknown"])
        labels.append(f"{meta['badge_emoji']} {meta['vn_name']}")
        val = prob_dict.get(cls, 0.0) * 100
        values.append(val)
        colors.append(meta["color_hex"])

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            cornerradius=6,
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(size=12, color='#E2E8F0', family='JetBrains Mono'),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#93C5FD')),
        xaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',
            ticksuffix="%",
            title="",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color='#CBD5E1'),
        ),
        margin=dict(l=10, r=20, t=36, b=10),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ==============================================================================
# HÀM CHÍNH RENDER ỨNG DỤNG
# ==============================================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 1. Hero Banner Header
    logo_path = PROJECT_ROOT / "assets" / "logo_uit.png"
    if not logo_path.exists():
        logo_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "logo_uit.png"
    col_logo, col_header = st.columns([1, 6])
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=100)
        else:
            st.markdown("<h1 style='text-align:center;'>🚦</h1>", unsafe_allow_html=True)
    with col_header:
        st.markdown(
            """
            <div class="hero-banner">
                <div class="hero-title">HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG VIỆT NAM</div>
                <div class="hero-subtitle">
                    Đồ án môn học: <b>Thị Giác Máy Tính (CS231)</b> — Khoa Khoa Học Máy Tính, Trường ĐH Công Nghệ Thông Tin (UIT - VNU-HCM)<br>
                    Quy chuẩn: <b>QCVN 41:2019/BGTVT</b> | Tiếp cận: <b>Học Máy Cổ Điển vs Học Sâu Tinh Gọn</b>
                </div>
                <div class="hero-tags">
                    <span class="hero-tag">🎯 Phân loại 4 nhóm chính</span>
                    <span class="hero-tag hero-tag-green">🧠 Enhanced LeNet-5 (0.56M params)</span>
                    <span class="hero-tag hero-tag-amber">📊 Color + HOG + SVM (457d)</span>
                    <span class="hero-tag">🏙️ Two-Stage YOLO11s Pipeline</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Nạp các model
    det_lenet = get_cached_lenet()
    det_svm = get_cached_svm()
    yolo_model = get_cached_yolo()

    # Sidebar điều khiển
    with st.sidebar:
        st.markdown("### ⚙️ Bảng Điều Khiển")
        test_mode = st.radio(
            "Chọn chế độ kiểm thử:",
            ["🖼️ Ảnh Cắt Sẵn (ROI Classification)", "🏙️ Toàn Cảnh Đường Phố (Two-Stage Pipeline)"],
            index=0,
        )

        chosen_model_name = st.selectbox(
            "Chọn thuật toán phân loại:",
            [
                "⚡ Enhanced LeNet-5 (Học Sâu - Thành viên 2)",
                "🎯 Color + HOG + SVM (Cổ Điển - Thành viên 1)",
                "⚔️ So Sánh Song Song Cả 2 Mô Hình (Side-by-Side)",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### 📊 Trạng Thái Checkpoints")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Enhanced LeNet-5:")
            st.caption("Color HOG SVM:")
            st.caption("YOLO11s Scene:")
        with c2:
            st.markdown("`" + ("✅ Sẵn sàng" if det_lenet else "❌ Thiếu") + "`")
            st.markdown("`" + ("✅ Sẵn sàng" if det_svm else "❌ Thiếu") + "`")
            st.markdown("`" + ("✅ Sẵn sàng" if yolo_model else "❌ Thiếu") + "`")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:12px; color:#94A3B8; line-height:1.6;">
                <b>👥 Phân Công Nhiệm Vụ:</b><br>
                • <b>Thành viên 1:</b> Phương pháp truyền thống (Color HSV + Lưới không gian + HOG + SVM RBF).<br>
                • <b>Thành viên 2:</b> Kiến trúc học sâu (Enhanced LeNet-5 + Global Average Pooling + YOLO11s).<br>
                <b>📚 Giảng Viên Hướng Dẫn:</b><br>
                • ThS. Huỳnh Hồ Thị Mộng Trinh
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Khởi tạo Tabs chính
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Thử Nghiệm Tương Tác",
        "🏆 Bảng Xếp Hạng Đối Chuẩn",
        "🧠 Kiến Trúc & Feature Maps",
        "📈 Khám Phá Dữ Liệu (EDA)",
    ])

    # ==========================================================================
    # TAB 1: THỬ NGHIỆM TƯƠNG TÁC
    # ==========================================================================
    with tab1:
        if "ROI" in test_mode:
            st.markdown("#### 1. Thử Nghiệm Trên Ảnh Cận Cảnh Biển Báo (ROI Test)")
            st.caption("Phân loại ảnh cắt vuông $64\\times 64$ vào 1 trong 4 nhóm quy chuẩn QCVN 41:2019.")

            # PRESET QUICK SELECT GALLERY
            st.markdown("##### ⚡ Bộ sưu tập ảnh mẫu kiểm thử nhanh (1-Click Preset Gallery):")
            p_cols = st.columns(4)
            preset_clicked = None
            for idx, (k, preset_info) in enumerate(PRESET_SAMPLES.items()):
                with p_cols[idx]:
                    if preset_info["path"].exists():
                        p_img = cv2.imread(str(preset_info["path"]))
                        p_img_rgb = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
                        st.image(p_img_rgb, caption=preset_info["label"], use_container_width=True)
                        if st.button(f"Chọn mẫu {preset_info['icon']}", key=f"btn_preset_{k}", use_container_width=True):
                            st.session_state["selected_roi_img"] = str(preset_info["path"])
                            st.session_state["selected_roi_class"] = preset_info["true_class"]

            st.markdown("---")
            col_input, col_pred = st.columns([1, 1.2])

            with col_input:
                uploaded_file = st.file_uploader(
                    "Hoặc tải lên ảnh biển báo của bạn (JPG, PNG, JPEG):",
                    type=["jpg", "jpeg", "png"],
                    key="roi_uploader",
                )

            # Xác định ảnh được chọn (từ preset hoặc từ upload)
            selected_bgr = None
            ground_truth_label = None

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                selected_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.pop("selected_roi_img", None)
            elif "selected_roi_img" in st.session_state:
                p_path = Path(st.session_state["selected_roi_img"])
                if p_path.exists():
                    selected_bgr = cv2.imread(str(p_path))
                    ground_truth_label = st.session_state.get("selected_roi_class")

            # Nếu có ảnh được chọn -> Thực hiện suy luận và hiển thị
            if selected_bgr is not None:
                selected_rgb = cv2.cvtColor(selected_bgr, cv2.COLOR_BGR2RGB)
                with col_input:
                    st.markdown("##### 🖼️ Ảnh Biển Báo Đầu Vào:")
                    st.image(selected_rgb, use_container_width=True)
                    if ground_truth_label:
                        gt_meta = CLASS_METADATA.get(ground_truth_label, CLASS_METADATA["Unknown"])
                        st.info(f"Nhãn thực tế (Ground Truth): **{gt_meta['badge_emoji']} {gt_meta['vn_name']} ({gt_meta['code_prefix']})**")

                with col_pred:
                    st.markdown("##### 🎯 Kết Quả Phân Loại:")

                    # Trường hợp 1: Enhanced LeNet-5
                    if "Enhanced LeNet-5" in chosen_model_name:
                        if det_lenet is None:
                            st.error("Chưa nạp được checkpoint LeNet-5 tại `checkpoints/deep_learning/best_cnn.pth`!")
                        else:
                            t0 = time.perf_counter()
                            res = det_lenet.predict(selected_bgr)
                            latency_ms = (time.perf_counter() - t0) * 1000
                            cname = res.get("class_name", "Unknown")
                            conf = res.get("confidence", 0.0)
                            meta = CLASS_METADATA.get(cname, CLASS_METADATA["Unknown"])

                            badge_class = {
                                "Bien_Cam": "badge-cam",
                                "Bien_Nguy_Hiem": "badge-nguy-hiem",
                                "Bien_Hieu_Lenh": "badge-hieu-lenh",
                                "Bien_Chi_Dan": "badge-chi-dan",
                            }.get(cname, "badge-unknown")

                            st.markdown(
                                f"""
                                <div class="result-badge {badge_class}">
                                    <span>{meta['badge_emoji']}</span>
                                    <span>{meta['vn_name']} ({meta['code_prefix']})</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.caption(f"📌 *{meta['description']}*")

                            # Metric boxes
                            st.markdown(
                                f"""
                                <div class="metric-container">
                                    <div class="metric-box">
                                        <div class="metric-value">{conf*100:.2f}%</div>
                                        <div class="metric-label">Độ Tin Cậy</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-value">{latency_ms:.2f} ms</div>
                                        <div class="metric-label">Độ Trễ Suy Luận</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-value">{1000/max(0.1, latency_ms):.0f}</div>
                                        <div class="metric-label">Tốc Độ (FPS)</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-value">0.56M</div>
                                        <div class="metric-label">Tham Số Mạng</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            probs = extract_probabilities(det_lenet, selected_bgr, "lenet")
                            plot_probability_bar_chart(probs, "Phân Phối Độ Tin Cậy (Softmax Logits)")

                    # Trường hợp 2: Color + HOG + SVM
                    elif "Color + HOG + SVM" in chosen_model_name:
                        if det_svm is None:
                            st.error("Chưa nạp được checkpoint SVM tại `checkpoints/color_hog_svm/stage2_best.joblib`!")
                        else:
                            t0 = time.perf_counter()
                            res = det_svm.predict(selected_bgr)
                            latency_ms = (time.perf_counter() - t0) * 1000
                            cname = res.get("class_name", "Unknown")
                            conf = res.get("confidence", 0.0)
                            meta = CLASS_METADATA.get(cname, CLASS_METADATA["Unknown"])

                            badge_class = {
                                "Bien_Cam": "badge-cam",
                                "Bien_Nguy_Hiem": "badge-nguy-hiem",
                                "Bien_Hieu_Lenh": "badge-hieu-lenh",
                                "Bien_Chi_Dan": "badge-chi-dan",
                            }.get(cname, "badge-unknown")

                            st.markdown(
                                f"""
                                <div class="result-badge {badge_class}">
                                    <span>{meta['badge_emoji']}</span>
                                    <span>{meta['vn_name']} ({meta['code_prefix']})</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.caption(f"📌 *{meta['description']}*")

                            # Metric boxes
                            st.markdown(
                                f"""
                                <div class="metric-container">
                                    <div class="metric-box">
                                        <div class="metric-value">{conf*100:.2f}%</div>
                                        <div class="metric-label">Độ Tin Cậy</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-value">{latency_ms:.2f} ms</div>
                                        <div class="metric-label">Độ Trễ Suy Luận</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-value">{1000/max(0.1, latency_ms):.0f}</div>
                                        <div class="metric-label">Tốc Độ (FPS)</div>
                                    </div>
                                    <div class="metric-box">
                                        <div class="metric-value">457d</div>
                                        <div class="metric-label">Số Chiều Đặc Trưng</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            probs = extract_probabilities(det_svm, selected_bgr, "svm")
                            plot_probability_bar_chart(probs, "Phân Phối Độ Tin Cậy (SVM Decision Softmax)")

                    # Trường hợp 3: Side-by-Side Duel Mode
                    else:
                        st.markdown("##### ⚔️ So Sánh Song Song 2 Phương Pháp:")
                        col_m1, col_m2 = st.columns(2)

                        with col_m1:
                            st.markdown("###### 👤 TV1: Color + HOG + SVM")
                            if det_svm is not None:
                                t0 = time.perf_counter()
                                r1 = det_svm.predict(selected_bgr)
                                lat1 = (time.perf_counter() - t0) * 1000
                                c1 = r1.get("class_name", "Unknown")
                                m1 = CLASS_METADATA.get(c1, CLASS_METADATA["Unknown"])
                                st.info(f"{m1['badge_emoji']} **{m1['vn_name']}**")
                                st.metric("Độ Tin Cậy", f"{r1.get('confidence', 0)*100:.1f}%")
                                st.caption(f"⏱️ Độ trễ: **{lat1:.2f} ms** | Không gian: **457 chiều**")
                                p1 = extract_probabilities(det_svm, selected_bgr, "svm")
                                plot_probability_bar_chart(p1, "SVM Probabilities")
                            else:
                                st.warning("Chưa có model SVM")

                        with col_m2:
                            st.markdown("###### 👤 TV2: Enhanced LeNet-5")
                            if det_lenet is not None:
                                t0 = time.perf_counter()
                                r2 = det_lenet.predict(selected_bgr)
                                lat2 = (time.perf_counter() - t0) * 1000
                                c2 = r2.get("class_name", "Unknown")
                                m2 = CLASS_METADATA.get(c2, CLASS_METADATA["Unknown"])
                                st.success(f"{m2['badge_emoji']} **{m2['vn_name']}**")
                                st.metric("Độ Tin Cậy", f"{r2.get('confidence', 0)*100:.1f}%")
                                st.caption(f"⏱️ Độ trễ: **{lat2:.2f} ms** | Tham số: **0.56M params**")
                                p2 = extract_probabilities(det_lenet, selected_bgr, "lenet")
                                plot_probability_bar_chart(p2, "LeNet-5 Probabilities")
                            else:
                                st.warning("Chưa có model LeNet")

            else:
                st.info("👆 Vui lòng bấm vào 1 trong các ảnh mẫu trên hoặc tải ảnh biển báo lên để xem kết quả phân loại thời gian thực.")

        else:
            # -------------------------------------------------------------
            # CHẾ ĐỘ TOÀN CẢNH ĐƯỜNG PHỐ (TWO-STAGE PIPELINE)
            # -------------------------------------------------------------
            st.markdown("#### 2. Thử Nghiệm Trên Ảnh Toàn Cảnh Đường Phố (Two-Stage Pipeline)")
            st.caption("Quy trình 2 tầng hoàn chỉnh: YOLO11s quét vị trí biển báo -> Crop ROI -> Enhanced LeNet-5 / SVM phân loại.")

            # Quick scene selector
            st.markdown("##### ⚡ Chọn ảnh chụp toàn cảnh mẫu:")
            sc_cols = st.columns(len(PRESET_SCENES))
            for idx, (sk, sinfo) in enumerate(PRESET_SCENES.items()):
                with sc_cols[idx]:
                    if sinfo["path"].exists():
                        if st.button(f"📷 {sinfo['label']}", key=f"btn_scene_{sk}", use_container_width=True):
                            st.session_state["selected_scene_img"] = str(sinfo["path"])

            scene_file = st.file_uploader(
                "Hoặc tải lên ảnh toàn cảnh đường phố (JPG, PNG):",
                type=["jpg", "jpeg", "png"],
                key="scene_uploader",
            )

            chosen_scene_bgr = None
            if scene_file is not None:
                file_bytes = np.asarray(bytearray(scene_file.read()), dtype=np.uint8)
                chosen_scene_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.pop("selected_scene_img", None)
            elif "selected_scene_img" in st.session_state:
                spath = Path(st.session_state["selected_scene_img"])
                if spath.exists():
                    chosen_scene_bgr = cv2.imread(str(spath))

            # Bộ điều khiển tham số YOLO
            c_conf, c_iou = st.columns(2)
            with c_conf:
                yolo_conf = st.slider("Ngưỡng tin cậy phát hiện (YOLO Confidence Threshold):", 0.10, 0.90, 0.25, 0.05)
            with c_iou:
                yolo_iou = st.slider("Ngưỡng giao nhau NMS (IoU Threshold):", 0.30, 0.80, 0.60, 0.05)

            if chosen_scene_bgr is not None:
                if yolo_model is None:
                    st.error("Không tìm thấy checkpoint YOLO11s tại `runs/detect/vn_s/weights/best.pt`!")
                else:
                    ih, iw = chosen_scene_bgr.shape[:2]
                    chosen_classifier = det_lenet if "Enhanced LeNet-5" in chosen_model_name else det_svm
                    if chosen_classifier is None:
                        chosen_classifier = det_lenet or det_svm

                    classifier_name = "Enhanced LeNet-5" if chosen_classifier == det_lenet else "Color + HOG + SVM"

                    with st.spinner("Đang thực hiện quy trình 2 tầng: YOLO11s phát hiện và phân loại..."):
                        t_start = time.perf_counter()
                        results = yolo_model.predict(
                            source=chosen_scene_bgr,
                            conf=yolo_conf,
                            iou=yolo_iou,
                            verbose=False,
                            imgsz=800,
                        )
                        t_yolo = (time.perf_counter() - t_start) * 1000

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

                        vis_bgr = chosen_scene_bgr.copy()
                        detected_signs = []
                        t_class_total = 0.0

                        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                            crop = chosen_scene_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                t_c0 = time.perf_counter()
                                pred = chosen_classifier.predict(crop)
                                t_class_total += (time.perf_counter() - t_c0) * 1000

                                cname = pred.get("class_name", "Unknown")
                                conf = pred.get("confidence", 0.0)
                                meta = CLASS_METADATA.get(cname, CLASS_METADATA["Unknown"])

                                # Vẽ bounding box đẹp
                                bgr_col = meta["color_bgr"]
                                cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), bgr_col, 3)

                                # Tag text nền đậm
                                tag_text = f"#{idx} {meta['code_prefix']} ({conf*100:.0f}%)"
                                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                                cv2.rectangle(vis_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), bgr_col, -1)
                                cv2.putText(
                                    vis_bgr,
                                    tag_text,
                                    (x1 + 3, max(th, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 255),
                                    2,
                                )

                                detected_signs.append({
                                    "STT": f"#{idx}",
                                    "Nhóm Dự Đoán": f"{meta['badge_emoji']} {meta['vn_name']}",
                                    "Mã Chuẩn": meta["code_prefix"],
                                    "Độ Tin Cậy": f"{conf*100:.1f}%",
                                    "Tọa Độ [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                                    "Kích Thước": f"{x2-x1} × {y2-y1} px",
                                })

                        t_total = t_yolo + t_class_total

                    # Hiển thị metrics toàn cảnh
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-box">
                                <div class="metric-value">{len(boxes)}</div>
                                <div class="metric-label">Biển Báo Phát Hiện</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{t_yolo:.1f} ms</div>
                                <div class="metric-label">YOLO11s Định Vị</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{t_class_total:.1f} ms</div>
                                <div class="metric-label">Phân Loại ({classifier_name})</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{1000/max(1.0, t_total):.1f} FPS</div>
                                <div class="metric-label">Tốc Độ Toàn Trình (E2E)</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_vis, col_tbl = st.columns([1.6, 1.2])
                    with col_vis:
                        st.image(
                            cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB),
                            caption=f"Ảnh toàn cảnh đã được nhận diện ({len(boxes)} biển báo)",
                            use_container_width=True,
                        )
                    with col_tbl:
                        st.markdown(f"##### 📋 Danh Sách Biển Báo Chi Tiết ({len(detected_signs)} biển):")
                        if detected_signs:
                            df_det = pd.DataFrame(detected_signs)
                            st.dataframe(df_det, use_container_width=True, hide_index=True)
                        else:
                            st.warning("Không tìm thấy biển báo nào trên khung hình với ngưỡng lọc hiện tại.")

    # ==========================================================================
    # TAB 2: ĐỐI CHUẨN HIỆU NĂNG TOÀN DIỆN (LEADERBOARDS)
    # ==========================================================================
    with tab2:
        st.markdown("### 🏆 Bảng Xếp Hạng Đối Chuẩn Toàn Diện (Official Benchmarks)")
        st.caption("Dữ liệu thực nghiệm chính thức từ Báo cáo Đồ án CS231 trên tập kiểm thử độc lập Blind Test (không rò rỉ dữ liệu).")

        st.markdown("#### 🥇 Tầng 1: Đánh Giá Năng Lực Phân Loại Trên Tập Blind Test (1.813 Mẫu ROI Cắt Sẵn)")

        df_bench_roi = pd.DataFrame([
            {
                "Mô Hình Phân Loại": "Enhanced LeNet-5 (Đề xuất)",
                "Loại Mô Hình": "Học Sâu (Deep Learning)",
                "Người Phụ Trách": "Thành viên 2",
                "Độ Chính Xác (%)": 98.73,
                "Macro F1 (%)": 95.66,
                "Độ Trễ (ms)": 1.10,
                "Tốc Độ (FPS)": 909.1,
                "Số Tham Số": "0.56M",
                "Dung Lượng (MB)": "2.2 MB",
            },
            {
                "Mô Hình Phân Loại": "Color + HOG + SVM (S3)",
                "Loại Mô Hình": "Học Máy Cổ Điển",
                "Người Phụ Trách": "Thành viên 1",
                "Độ Chính Xác (%)": 98.57,
                "Macro F1 (%)": 95.51,
                "Độ Trễ (ms)": 2.63,
                "Tốc Độ (FPS)": 380.2,
                "Số Tham Số": "457 chiều",
                "Dung Lượng (MB)": "14.8 MB",
            },
            {
                "Mô Hình Phân Loại": "Classical LeNet-5 (Baseline)",
                "Loại Mô Hình": "Học Sâu Cổ Điển",
                "Người Phụ Trách": "Baseline",
                "Độ Chính Xác (%)": 96.25,
                "Macro F1 (%)": 91.80,
                "Độ Trễ (ms)": 0.85,
                "Tốc Độ (FPS)": 1176.5,
                "Số Tham Số": "0.06M",
                "Dung Lượng (MB)": "0.3 MB",
            },
            {
                "Mô Hình Phân Loại": "MobileNetV3-Small",
                "Loại Mô Hình": "Học Sâu Hiện Đại",
                "Người Phụ Trách": "Tham Chiếu",
                "Độ Chính Xác (%)": 98.62,
                "Macro F1 (%)": 95.40,
                "Độ Trễ (ms)": 2.80,
                "Tốc Độ (FPS)": 357.1,
                "Số Tham Số": "1.52M",
                "Dung Lượng (MB)": "6.1 MB",
            },
            {
                "Mô Hình Phân Loại": "ResNet-18",
                "Loại Mô Hình": "Học Sâu Hiện Đại",
                "Người Phụ Trách": "Tham Chiếu",
                "Độ Chính Xác (%)": 98.84,
                "Macro F1 (%)": 95.80,
                "Độ Trễ (ms)": 4.25,
                "Tốc Độ (FPS)": 235.3,
                "Số Tham Số": "11.17M",
                "Dung Lượng (MB)": "44.7 MB",
            },
        ])

        st.dataframe(
            df_bench_roi.style.highlight_max(subset=["Độ Chính Xác (%)", "Macro F1 (%)", "Tốc Độ (FPS)"], color="#1E3A8A"),
            use_container_width=True,
            hide_index=True,
        )

        # Plotly Pareto Tradeoff: Accuracy vs Latency
        if HAS_PLOTLY:
            fig_tradeoff = px.scatter(
                df_bench_roi,
                x="Độ Trễ (ms)",
                y="Độ Chính Xác (%)",
                text="Mô Hình Phân Loại",
                size=[18, 14, 10, 14, 20],
                color="Loại Mô Hình",
                color_discrete_map={
                    "Học Sâu (Deep Learning)": "#38BDF8",
                    "Học Máy Cổ Điển": "#F59E0B",
                    "Học Sâu Cổ Điển": "#9CA3AF",
                    "Học Sâu Hiện Đại": "#10B981",
                },
                title="Đánh Đổi Hiệu Năng: Độ Chính Xác (Accuracy) vs Thời Gian Suy Luận (Latency)",
            )
            fig_tradeoff.update_traces(textposition='top center')
            fig_tradeoff.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#CBD5E1"),
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', title="Độ trễ suy luận (ms) - Càng thấp càng tốt"),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', title="Độ chính xác (%) - Càng cao càng tốt", range=[95, 100]),
                height=380,
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🥈 Tầng 2: Đánh Giá Toàn Trình Trên Ảnh Đường Phố Toàn Cảnh (End-to-End Scene Benchmark)")
        st.caption("Kiểm nghiệm hệ thống phát hiện 2 giai đoạn trên toàn bộ 3.216 ảnh đường phố độ phân giải cao.")

        df_bench_scene = pd.DataFrame([
            {
                "Pipeline Toàn Trình": "Two-Stage LeNet (YOLO11s + Enhanced LeNet-5)",
                "Tỷ Lệ Định Vị Đúng (IoU >= 0.5)": "85.61%",
                "Đoán Đúng Khi Đã Định Vị": "99.40%",
                "Recall Toàn Trình (E2E)": "85.10%",
                "Độ Trễ Toàn Cảnh": "91.45 ms",
                "Tốc Độ Xử Lý": "10.9 FPS",
            },
            {
                "Pipeline Toàn Trình": "Two-Stage SVM (YOLO11s + Color HOG SVM)",
                "Tỷ Lệ Định Vị Đúng (IoU >= 0.5)": "85.61%",
                "Đoán Đúng Khi Đã Định Vị": "99.17%",
                "Recall Toàn Trình (E2E)": "84.90%",
                "Độ Trễ Toàn Cảnh": "105.20 ms",
                "Tốc Độ Xử Lý": "9.5 FPS",
            },
        ])
        st.dataframe(df_bench_scene, use_container_width=True, hide_index=True)

        # Confusion Matrices comparison
        st.markdown("#### 🔍 Ma Trận Nhầm Lẫn Đối Chuẩn (Confusion Matrices)")
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            cm_lenet_path = PROJECT_ROOT / "assets" / "confusion_matrix_deep_learning.png"
            if not cm_lenet_path.exists():
                cm_lenet_path = RESULTS_DIR / "confusion_matrix_deep_learning.png"
            if not cm_lenet_path.exists():
                cm_lenet_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "sec4" / "f3_03_lenet_confusion_matrix.png"
            if cm_lenet_path.exists():
                st.image(str(cm_lenet_path), caption="Ma trận nhầm lẫn: Enhanced LeNet-5 (Thành viên 2)", use_container_width=True)

        with cm_col2:
            cm_svm_path = PROJECT_ROOT / "assets" / "confusion_matrix_color_hog_svm.png"
            if not cm_svm_path.exists():
                cm_svm_path = RESULTS_DIR / "confusion_matrix_color_hog_svm.png"
            if not cm_svm_path.exists():
                cm_svm_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "sec4" / "f1_17_confusion_matrices_levels.png"
            if cm_svm_path.exists():
                st.image(str(cm_svm_path), caption="Ma trận nhầm lẫn: Color + HOG + SVM (Thành viên 1)", use_container_width=True)

    # ==========================================================================
    # TAB 3: KIẾN TRÚC MẠNG & GIẢI THÍCH (EXPLAINABILITY & FEATURE MAPS)
    # ==========================================================================
    with tab3:
        st.markdown("### 🧠 Kiến Trúc Mô Hình & Trực Quan Hóa Feature Maps (Explainability)")
        st.caption("Khám phá cơ chế trích xuất đặc trưng qua từng tầng tích chập và lý do thiết kế kiến trúc.")

        tab_arch_dl, tab_arch_trad = st.tabs(["🚀 Enhanced LeNet-5 (Học Sâu)", "⚙️ Color + HOG + SVM (Cổ Điển)"])

        with tab_arch_dl:
            st.markdown("#### 1. Sơ Đồ Kiến Trúc Mạng Enhanced LeNet-5")
            arch_img_path = PROJECT_ROOT / "assets" / "f3_01_lenet_architecture.png"
            if not arch_img_path.exists():
                arch_img_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "sec3" / "f3_01_lenet_architecture.png"
            if arch_img_path.exists():
                st.image(str(arch_img_path), caption="Kiến trúc Enhanced LeNet-5 với 3 Conv Blocks, BatchNorm, GAP và Dropout", use_container_width=True)

            st.markdown("#### 2. Trực Quan Hóa Feature Maps (Kích Hoạt Qua Từng Tầng Conv)")
            fmap_img_path = PROJECT_ROOT / "assets" / "f3_04_lenet_feature_maps.png"
            if not fmap_img_path.exists():
                fmap_img_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "sec3" / "f3_04_lenet_feature_maps.png"
            if fmap_img_path.exists():
                st.image(str(fmap_img_path), caption="Bản đồ kích hoạt: Tầng Conv1 bắt viền/màu; Tầng Conv3 bắt ngữ nghĩa biểu tượng", use_container_width=True)

            st.markdown("#### 3. Các Điểm Sáng Tạo Kiến Trúc (Architecture Insights):")
            c_ins1, c_ins2, c_ins3 = st.columns(3)
            with c_ins1:
                st.markdown(
                    """
                    <div class="glass-card">
                        <h4 style="color:#38BDF8; margin-top:0;">🖼️ Độ Phân Giải 64×64</h4>
                        <p style="font-size:13px; color:#CBD5E1; line-height:1.5;">
                            Giữ lại cấu trúc chi tiết của các biển báo phức tạp (mũi tên, số km/h, xe cộ) thay vì nén quá mức về 32×32 như LeNet-5 nguyên bản năm 1998.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_ins2:
                st.markdown(
                    """
                    <div class="glass-card">
                        <h4 style="color:#10B981; margin-top:0;">⚡ Global Average Pooling</h4>
                        <p style="font-size:13px; color:#CBD5E1; line-height:1.5;">
                            Thay thế toàn bộ lớp Flatten cồng kềnh bằng GAP (128 giá trị trung bình), giảm <b>88.3%</b> số lượng trọng số FC, triệt tiêu nguy cơ học vẹt (overfitting).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_ins3:
                st.markdown(
                    """
                    <div class="glass-card">
                        <h4 style="color:#F59E0B; margin-top:0;">🏎️ Tối Ưu Thiết Bị Nhúng</h4>
                        <p style="font-size:13px; color:#CBD5E1; line-height:1.5;">
                            Chỉ <b>0.56M tham số</b> (2.2 MB), tốc độ đạt <b>909 FPS</b>, nhanh gấp 3.8 lần ResNet-18 mà độ chính xác chênh lệch dưới 0.11%, cực kỳ lý tưởng cho Edge AI camera.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_arch_trad:
            st.markdown("#### 1. Cấu Trúc Vector Đặc Trưng Kết Hợp S3 (457 Chiều)")
            feat_struct_path = PROJECT_ROOT / "assets" / "f1_14_feature_vector_structure.png"
            if not feat_struct_path.exists():
                feat_struct_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "sec3" / "f1_14_feature_vector_structure.png"
            if feat_struct_path.exists():
                st.image(str(feat_struct_path), caption="Sơ đồ phân rã Vector đặc trưng 457 chiều: Màu sắc HSV + Lưới không gian + HOG", use_container_width=True)

            st.markdown("#### 2. Trực Quan Hóa HOG (Histogram of Oriented Gradients) Theo Nhóm Biển")
            hog_vis_path = PROJECT_ROOT / "assets" / "f1_12_hog_visualization_groups.png"
            if not hog_vis_path.exists():
                hog_vis_path = PROJECT_ROOT / "docs" / "CS231_tex_package" / "graphics" / "sec3" / "f1_12_hog_visualization_groups.png"
            if hog_vis_path.exists():
                st.image(str(hog_vis_path), caption="Biểu diễn hướng gradient (HOG) thể hiện biên tròn, tam giác và hình chữ nhật", use_container_width=True)

    # ==========================================================================
    # TAB 4: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
    # ==========================================================================
    with tab4:
        st.markdown("### 📈 Phân Tích Dữ Liệu Khám Phá (Exploratory Data Analysis - EDA)")
        st.markdown(
            "Tập dữ liệu gồm **3.216 ảnh chụp đường phố** tại Việt Nam chứa **8.334 bounding boxes**, "
            "được phân loại vào 4 nhóm đối tượng chính theo quy chuẩn quốc gia **QCVN 41:2019/BGTVT**."
        )

        eda_p1 = PROJECT_ROOT / "assets" / "eda" / "1_class_distribution.png"
        if not eda_p1.exists():
            eda_p1 = RESULTS_DIR / "eda" / "1_class_distribution.png"

        eda_p2 = PROJECT_ROOT / "assets" / "eda" / "2_bbox_dimensions.png"
        if not eda_p2.exists():
            eda_p2 = RESULTS_DIR / "eda" / "2_bbox_dimensions.png"

        eda_p3 = PROJECT_ROOT / "assets" / "eda" / "3_sample_traffic_signs_grid.png"
        if not eda_p3.exists():
            eda_p3 = RESULTS_DIR / "eda" / "3_sample_traffic_signs_grid.png"

        eda_p4 = PROJECT_ROOT / "assets" / "eda" / "4_split_distribution.png"
        if not eda_p4.exists():
            eda_p4 = RESULTS_DIR / "eda" / "4_split_distribution.png"

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            if eda_p1.exists():
                st.image(str(eda_p1), caption="Phân bố số lượng mẫu qua 4 nhóm biển báo mục tiêu", use_container_width=True)
            if eda_p3.exists():
                st.image(str(eda_p3), caption="Lưới ảnh mẫu các biển báo thực tế thu thập tại Việt Nam", use_container_width=True)

        with col_eda2:
            if eda_p2.exists():
                st.image(str(eda_p2), caption="Phân bố kích thước và tỉ lệ khung hình (Aspect Ratio)", use_container_width=True)
            if eda_p4.exists():
                st.image(str(eda_p4), caption="Tỉ lệ phân chia tập dữ liệu (Train 70% / Val 15% / Test 15%)", use_container_width=True)


if __name__ == "__main__":
    main()
