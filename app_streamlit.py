"""
Streamlit Interactive Demo & Benchmarking Dashboard (QCVN 41:2019/BGTVT).
Run: venv/Scripts/python.exe -m streamlit run app_streamlit.py
"""

import logging
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, TARGET_CLASSES
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.evaluation.scene_eval import load_yolo_detector

from demo_ui.charts import (
    HAS_PLOTLY,
    build_fps_bar_chart,
    build_probability_bar_chart,
    build_tradeoff_scatter,
)
from demo_ui.constants import (
    BENCHMARK_ROI_ROWS,
    BENCHMARK_SCENE_ROWS,
    CLASS_METADATA,
    MODEL_OPTIONS,
    PRESET_ROIS,
    PRESET_SCENES,
)
from demo_ui.inference import (
    build_detections_dataframe,
    classify_single_crop,
    decode_image_bytes,
    draw_scene_boxes,
    load_image_path,
    resize_for_display,
    run_scene_inference,
    to_crops_zip_bytes,
    to_csv_bytes,
    to_png_bytes,
    upscale_crop_for_display,
)
from demo_ui.styles import (
    get_base_css,
    render_checkpoint_rows_html,
    render_header_html,
    render_qcvn_pill_html,
)

st.set_page_config(
    page_title="Nhận Dạng Biển Báo Giao Thông (CS231 - UIT)",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_cached_lenet():
    """Load and cache Enhanced LeNet-5 checkpoint once."""
    try:
        det = DeepLearningDetector()
        return det if det.model is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Không nạp được Enhanced LeNet-5")
        return None


@st.cache_resource(show_spinner=False)
def get_cached_svm():
    """Load and cache Color + HOG + SVM checkpoint once."""
    try:
        det = ColorHogSvmDetector()
        return det if det.model is not None else None
    except Exception:
        logging.getLogger(__name__).exception("Không nạp được Color + HOG + SVM")
        return None


@st.cache_resource(show_spinner=False)
def get_cached_yolo():
    """Load and cache YOLO11s 1-class detector checkpoint once."""
    try:
        return load_yolo_detector()
    except Exception:
        logging.getLogger(__name__).exception("Không nạp được YOLO11s")
        return None


def render_prob_chart(prob_dict: dict, title: str, key: str):
    """Render Plotly probability bar chart with native st.bar_chart fallback."""
    fig = build_probability_bar_chart(prob_dict, title=title)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
    else:
        df_fb = pd.DataFrame(
            {
                "Nhóm biển": [CLASS_METADATA[c]["vn_name"] for c in TARGET_CLASSES],
                "Xác suất (%)": [prob_dict.get(c, 0.0) * 100.0 for c in TARGET_CLASSES],
            }
        ).set_index("Nhóm biển")
        st.bar_chart(df_fb)


def render_probability_note():
    st.caption(
        "Ghi chú xác suất: % của **Enhanced LeNet-5** là xác suất Softmax từ 4 logits đầu ra; "
        "% của **Color + HOG + SVM** là Softmax tính trên `decision_function` 5 lớp OvR "
        "(gồm `phụ` cộng vào `Bien_Chi_Dan` theo `LABEL_MAPPING`), chưa hiệu chỉnh Platt Scaling nên chỉ dùng để tham khảo tương đối."
    )


def render_roi_classification_result(
    img_bgr,
    det_lenet,
    det_svm,
    model_key: str,
    prefix_key: str = "roi",
):
    """Render ROI classification result (single model or LeNet vs SVM comparison)."""
    res = classify_single_crop(
        crop_bgr=img_bgr,
        det_lenet=det_lenet,
        det_svm=det_svm,
        model_key=model_key,
        include_both=(model_key == "compare"),
    )

    if model_key == "compare":
        col_l, col_s = st.columns(2)
        with col_l:
            with st.container(border=True):
                st.markdown("##### Enhanced LeNet-5 (CNN)")
                if "lenet" in res:
                    r1 = res["lenet"]
                    st.markdown(render_qcvn_pill_html(r1["class_name"]), unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Độ tin cậy", f"{r1['confidence']*100:.2f}%")
                    m2.metric("Độ trễ", f"{r1['latency_ms']:.2f} ms")
                    render_prob_chart(r1["probs"], "Phân phối Softmax (LeNet-5)", key=f"{prefix_key}_cmp_lenet")
                else:
                    st.warning("Thiếu checkpoint `best_cnn.pth`.")
        with col_s:
            with st.container(border=True):
                st.markdown("##### Color + HOG + SVM")
                if "svm" in res:
                    r2 = res["svm"]
                    st.markdown(render_qcvn_pill_html(r2["class_name"]), unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Độ tin cậy", f"{r2['confidence']*100:.2f}%")
                    m2.metric("Độ trễ", f"{r2['latency_ms']:.2f} ms")
                    render_prob_chart(r2["probs"], "Decision Softmax (SVM)", key=f"{prefix_key}_cmp_svm")
                else:
                    st.warning("Thiếu checkpoint `stage2_best.joblib`.")
    else:
        sub_key = "lenet" if model_key == "lenet" else "svm"
        if sub_key not in res:
            st.error("Chưa nạp được checkpoint mô hình đã chọn.")
            return
        r = res[sub_key]
        meta = CLASS_METADATA.get(r["class_name"], CLASS_METADATA["Unknown"])
        with st.container(border=True):
            st.markdown(render_qcvn_pill_html(r["class_name"]), unsafe_allow_html=True)
            st.caption(meta["description"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Độ tin cậy", f"{r['confidence']*100:.2f}%")
            c2.metric("Độ trễ suy luận", f"{r['latency_ms']:.2f} ms")
            c3.metric("Thông lượng ước tính", f"{1000.0 / max(0.1, r['latency_ms']):.0f} FPS")
            chart_title = "Phân phối Softmax (LeNet-5)" if sub_key == "lenet" else "Decision Softmax (SVM)"
            render_prob_chart(r["probs"], chart_title, key=f"{prefix_key}_single_{sub_key}")

    render_probability_note()


def render_tab_interactive(det_lenet, det_svm, yolo_model, input_mode: str, model_key: str, yolo_conf: float, yolo_iou: float):
    """Tab 1: Unified image upload/preset demo (Two-Stage scene default + optional ROI mode)."""
    if input_mode == "scene":
        st.subheader("Thử nghiệm phân tích ảnh bất kỳ (YOLO11s + Phân loại)")
        st.caption("Tải lên ảnh đường phố (hoặc chọn ảnh mẫu): YOLO11s định vị biển báo -> cắt crop gốc -> phân loại 4 nhóm QCVN 41.")

        st.markdown("**Ảnh mẫu kiểm thử nhanh:**")
        p_cols = st.columns(len(PRESET_SCENES))
        for idx, (skey, sinfo) in enumerate(PRESET_SCENES.items()):
            with p_cols[idx]:
                exists = sinfo["path"].exists()
                if st.button(sinfo["label"], key=f"btn_scene_{skey}", width="stretch", disabled=not exists):
                    st.session_state["active_scene_path"] = str(sinfo["path"])
                    st.session_state["active_scene_name"] = sinfo["path"].name
                    st.session_state["force_roi_fallback"] = False

        uploaded = st.file_uploader(
            "Hoặc tải lên ảnh từ máy (JPG, JPEG, PNG — tự động xử lý xoay EXIF & kênh Alpha PNG):",
            type=["jpg", "jpeg", "png"],
            key="scene_file_uploader",
        )

        img_bgr = None
        img_info = {}
        source_name = ""

        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            if st.session_state.get("_last_upload_name") != uploaded.name or st.session_state.get("_last_upload_len") != len(raw_bytes):
                st.session_state["force_roi_fallback"] = False
                st.session_state["_last_upload_name"] = uploaded.name
                st.session_state["_last_upload_len"] = len(raw_bytes)
            img_bgr, img_info = decode_image_bytes(raw_bytes)
            source_name = uploaded.name
        elif "active_scene_path" in st.session_state:
            spath = Path(st.session_state["active_scene_path"])
            img_bgr, img_info = load_image_path(spath)
            source_name = st.session_state.get("active_scene_name", spath.name)

        if img_bgr is None:
            if uploaded is not None:
                st.error("Không thể giải mã file ảnh vừa tải lên. Vui lòng thử ảnh JPG hoặc PNG hợp lệ.")
            else:
                st.info("Chọn một ảnh mẫu phía trên hoặc tải lên một ảnh bất kỳ để chạy thử pipeline.")
            return

        if yolo_model is None:
            st.error("Thiếu checkpoint YOLO11s tại `checkpoints/yolo/best.pt` hoặc `runs/detect/vn_s/weights/best.pt`.")
            return

        iw, ih = img_info["width"], img_info["height"]
        notices = [f"Ảnh `{source_name}` ({iw} × {ih} px, gốc {img_info.get('orig_mode', 'RGB')})"]
        if img_info.get("exif_rotated"):
            notices.append(f"Đã tự động xoay theo thẻ EXIF (Orientation={img_info['exif_orientation']})")
        if img_info.get("had_alpha"):
            notices.append("Đã tách kênh Alpha PNG trên nền trắng")
        st.caption(" · ".join(notices))

        with st.spinner("Đang chạy YOLO11s định vị và phân loại biển báo..."):
            scene_out = run_scene_inference(
                img_bgr=img_bgr,
                yolo_model=yolo_model,
                det_lenet=det_lenet,
                det_svm=det_svm,
                model_key=model_key,
                conf_thresh=yolo_conf,
                iou_thresh=yolo_iou,
            )

        detections = scene_out["detections"]

        # Case: 0 signs detected
        if not detections:
            st.warning(
                f"YOLO11s không phát hiện biển báo nào ở ngưỡng tin cậy `conf = {yolo_conf:.2f}`. "
                "Bạn có thể hạ thanh trượt **Ngưỡng tin cậy YOLO** ở thanh bên để phát hiện biển nhỏ/mờ."
            )
            ratio = iw / max(1.0, float(ih))
            is_roi_like = max(iw, ih) <= 320 or (0.55 <= ratio <= 1.85 and max(iw, ih) <= 640)
            if is_roi_like:
                st.info(
                    "Ảnh này có kích thước nhỏ hoặc gần vuông, trông giống ảnh một biển báo đã cắt sẵn. "
                    "Bạn có thể phân loại trực tiếp toàn bộ ảnh như 1 ROI bên dưới."
                )

            col_prev, col_fb = st.columns([1, 1.3])
            with col_prev:
                disp_rgb, was_scaled = resize_for_display(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                st.image(disp_rgb, caption=f"Ảnh gốc ({iw} × {ih} px)", width="stretch")
                if was_scaled:
                    st.caption("Ảnh hiển thị đã được thu gọn (suy luận vẫn dùng độ phân giải gốc).")
            with col_fb:
                if st.button(
                    "Phân loại cả ảnh như 1 biển đã cắt (ROI)",
                    key="btn_fallback_roi",
                    type="primary",
                    icon=":material/crop_free:",
                ):
                    st.session_state["force_roi_fallback"] = True
                if st.session_state.get("force_roi_fallback"):
                    render_roi_classification_result(
                        img_bgr=img_bgr,
                        det_lenet=det_lenet,
                        det_svm=det_svm,
                        model_key=model_key,
                        prefix_key="fallback_roi",
                    )
            return

        # Case: >= 1 detections
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Biển báo phát hiện", f"{len(detections)}")
        m2.metric("YOLO11s định vị", f"{scene_out['yolo_ms']:.1f} ms")
        m3.metric(f"Phân loại ({MODEL_OPTIONS[model_key]})", f"{scene_out['cls_total_ms']:.1f} ms")
        m4.metric("Tốc độ toàn trình", f"{scene_out['fps']:.1f} FPS")

        annotated_bgr = draw_scene_boxes(img_bgr, detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        disp_rgb, was_scaled = resize_for_display(annotated_rgb)

        st.image(
            disp_rgb,
            caption=f"Kết quả phát hiện & phân loại trên {source_name} ({len(detections)} biển báo)",
            width="stretch",
        )
        if was_scaled:
            st.caption("Ảnh hiển thị được thu gọn cho vừa khung nhìn; toàn bộ suy luận & file tải về sử dụng ảnh độ phân giải gốc.")

        # Download buttons
        d1, d2, d3 = st.columns(3)
        stem = Path(source_name).stem or "result"
        d1.download_button(
            "Ảnh đã khoanh (PNG)",
            data=to_png_bytes(annotated_bgr),
            file_name=f"{stem}_annotated.png",
            mime="image/png",
            width="stretch",
            icon=":material/image:",
        )
        d2.download_button(
            "Bảng kết quả (CSV)",
            data=to_csv_bytes(detections, model_key=model_key),
            file_name=f"{stem}_detections.csv",
            mime="text/csv",
            width="stretch",
            icon=":material/table_view:",
        )
        d3.download_button(
            "Các ảnh crop (ZIP)",
            data=to_crops_zip_bytes(detections),
            file_name=f"{stem}_crops.zip",
            mime="application/zip",
            width="stretch",
            icon=":material/folder_zip:",
        )

        # Crop grid (raw crops passed to classifier; upscaled strictly for display)
        st.markdown("#### Lưới ảnh crop từng biển (đầu vào gốc đưa vào bộ phân loại)")
        grid_cols = 4
        for start in range(0, len(detections), grid_cols):
            cols = st.columns(grid_cols)
            for offset, det in enumerate(detections[start : start + grid_cols]):
                with cols[offset]:
                    with st.container(border=True):
                        x1, y1, x2, y2 = det["bbox"]
                        crop_rgb = cv2.cvtColor(det["crop_bgr"], cv2.COLOR_BGR2RGB)
                        crop_disp = upscale_crop_for_display(crop_rgb, target_min=140)
                        st.image(crop_disp, width="stretch")
                        st.markdown(
                            render_qcvn_pill_html(det["primary_class"], f"{det['primary_conf']*100:.1f}%"),
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='crop-meta'>#{det['index']} · [{x1}, {y1}, {x2}, {y2}]<br>{det['width']}×{det['height']} px · YOLO {det['yolo_conf']*100:.1f}%</div>",
                            unsafe_allow_html=True,
                        )

        # Probability distribution per detected sign + comparison toggle
        st.markdown("#### Phân tích xác suất 4 nhóm cho từng biển")
        sel_options = [
            f"Biển #{d['index']} — {CLASS_METADATA.get(d['primary_class'], CLASS_METADATA['Unknown'])['vn_name']} ({d['width']}×{d['height']} px)"
            for d in detections
        ]
        c_sel, c_cmp = st.columns([2, 1.2])
        with c_sel:
            chosen_idx = st.selectbox(
                "Chọn biển báo để xem chi tiết:",
                options=list(range(len(detections))),
                format_func=lambda i: sel_options[i],
                key="selected_det_idx",
            )
        with c_cmp:
            compare_selected = st.checkbox(
                "So sánh LeNet-5 và SVM cho biển này",
                value=(model_key == "compare"),
                key="compare_selected_crop",
                disabled=(det_lenet is None or det_svm is None),
            )

        sel_det = detections[chosen_idx]
        if compare_selected or model_key == "compare":
            comp_res = classify_single_crop(
                crop_bgr=sel_det["crop_bgr"],
                det_lenet=det_lenet,
                det_svm=det_svm,
                model_key="compare",
                include_both=True,
            )
            col_l, col_s = st.columns(2)
            with col_l:
                with st.container(border=True):
                    r_l = comp_res.get("lenet")
                    if r_l:
                        st.markdown(f"**Biển #{sel_det['index']} · Enhanced LeNet-5** ({r_l['confidence']*100:.1f}%)")
                        render_prob_chart(r_l["probs"], "Softmax Logits (LeNet-5)", key=f"det_lenet_{sel_det['index']}")
            with col_s:
                with st.container(border=True):
                    r_s = comp_res.get("svm")
                    if r_s:
                        st.markdown(f"**Biển #{sel_det['index']} · Color + HOG + SVM** ({r_s['confidence']*100:.1f}%)")
                        render_prob_chart(r_s["probs"], "Decision Softmax (SVM)", key=f"det_svm_{sel_det['index']}")
        else:
            sub_title = "Softmax Logits (LeNet-5)" if model_key == "lenet" else "Decision Softmax (SVM)"
            render_prob_chart(
                sel_det["primary_probs"],
                f"Biển #{sel_det['index']} · {sub_title}",
                key=f"det_single_{sel_det['index']}_{model_key}",
            )
        render_probability_note()

        # Structured results table
        st.markdown("#### Bảng tổng hợp kết quả từng biển báo")
        df_det = build_detections_dataframe(detections, model_key=model_key)
        st.dataframe(df_det, width="stretch", hide_index=True)

    else:
        # Optional standalone ROI mode
        st.subheader("Phân loại ảnh biển báo đã cắt sẵn (ROI Mode)")
        st.caption("Chế độ phụ: bỏ qua bước YOLO11s, đưa trực tiếp ảnh cận cảnh một biển báo vào bộ phân loại.")

        p_cols = st.columns(len(PRESET_ROIS))
        for idx, (rkey, rinfo) in enumerate(PRESET_ROIS.items()):
            with p_cols[idx]:
                with st.container(border=True):
                    if rinfo["path"].exists():
                        p_bgr, _ = load_image_path(rinfo["path"])
                        if p_bgr is not None:
                            p_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
                            st.image(upscale_crop_for_display(p_rgb, target_min=120), width="stretch")
                        if st.button(rinfo["label"], key=f"btn_roi_{rkey}", width="stretch"):
                            st.session_state["active_roi_path"] = str(rinfo["path"])
                            st.session_state["active_roi_gt"] = rinfo["true_class"]

        roi_upload = st.file_uploader(
            "Hoặc tải lên ảnh cận cảnh một biển báo (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            key="roi_only_uploader",
        )

        roi_bgr = None
        roi_gt = None
        if roi_upload is not None:
            roi_bgr, _ = decode_image_bytes(roi_upload.getvalue())
        elif "active_roi_path" in st.session_state:
            rpath = Path(st.session_state["active_roi_path"])
            roi_bgr, _ = load_image_path(rpath)
            roi_gt = st.session_state.get("active_roi_gt")

        if roi_bgr is None:
            st.info("Chọn một mẫu ROI phía trên hoặc tải lên ảnh một biển báo để phân loại.")
            return

        c_img, c_res = st.columns([1, 1.5])
        with c_img:
            with st.container(border=True):
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                st.image(upscale_crop_for_display(roi_rgb, target_min=180), caption=f"Kích thước gốc: {roi_bgr.shape[1]} × {roi_bgr.shape[0]} px", width="stretch")
                if roi_gt:
                    gt_meta = CLASS_METADATA.get(roi_gt, CLASS_METADATA["Unknown"])
                    st.caption(f"Nhãn thực tế (Ground Truth): **{gt_meta['vn_name']} ({gt_meta['code_prefix']})**")
        with c_res:
            render_roi_classification_result(
                img_bgr=roi_bgr,
                det_lenet=det_lenet,
                det_svm=det_svm,
                model_key=model_key,
                prefix_key="roi_mode",
            )


def render_tab_benchmarks():
    """Tab 2: Official benchmarks with non-overlapping charts & preserved table numbers."""
    st.subheader("Bảng Xếp Hạng Đối Chuẩn (Benchmarks)")
    st.caption("Số liệu giữ nguyên theo báo cáo trong repo để đảm bảo tính nhất quán khi nghiệm thu.")

    st.markdown("#### Tầng 1: Phân loại trên tập Blind Test ROI (1.813 mẫu cắt sẵn)")
    df_roi = pd.DataFrame(BENCHMARK_ROI_ROWS)
    st.dataframe(
        df_roi,
        width="stretch",
        hide_index=True,
        column_config={
            "Độ Chính Xác (%)": st.column_config.NumberColumn(format="%.2f"),
            "Macro F1 (%)": st.column_config.NumberColumn(format="%.2f"),
            "Độ Trễ (ms)": st.column_config.NumberColumn(format="%.2f"),
            "Tốc Độ (FPS)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    if HAS_PLOTLY:
        fig_scatter = build_tradeoff_scatter(df_roi)
        st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False}, key="bench_scatter")
        st.caption(
            "Lưu ý biểu đồ: Trục Y bắt đầu từ 95% để quan sát rõ vùng hội tụ; "
            "kích thước điểm đánh dấu cố định (chỉ phân biệt mô hình bằng màu và ký hiệu, rê chuột vào điểm để xem đủ chỉ số)."
        )
        fig_fps = build_fps_bar_chart(df_roi)
        st.plotly_chart(fig_fps, width="stretch", config={"displayModeBar": False}, key="bench_fps")
    else:
        st.bar_chart(df_roi.set_index("Mô Hình Phân Loại")[["Độ Chính Xác (%)", "Macro F1 (%)"]])

    st.divider()
    st.markdown("#### Tầng 2: Đánh giá toàn trình trên ảnh đường phố (End-to-End Scene)")
    st.caption("Tập kiểm thử End-to-End trong báo cáo: 550 ảnh Blind Test (lưu ý: nhãn gốc trong app cũ từng ghi 3.216 ảnh là tổng toàn bộ dataset).")
    df_scene = pd.DataFrame(BENCHMARK_SCENE_ROWS)
    st.dataframe(df_scene, width="stretch", hide_index=True)

    st.markdown("#### Ma trận nhầm lẫn đối chuẩn (Confusion Matrices)")
    cm_c1, cm_c2 = st.columns(2)
    with cm_c1:
        p_cm_l = PROJECT_ROOT / "assets" / "confusion_matrix_deep_learning.png"
        if not p_cm_l.exists():
            p_cm_l = RESULTS_DIR / "confusion_matrix_deep_learning.png"
        if p_cm_l.exists():
            st.image(str(p_cm_l), caption="Enhanced LeNet-5 (Thành viên 2)", width="stretch")
    with cm_c2:
        p_cm_s = PROJECT_ROOT / "assets" / "confusion_matrix_color_hog_svm.png"
        if not p_cm_s.exists():
            p_cm_s = RESULTS_DIR / "confusion_matrix_color_hog_svm.png"
        if p_cm_s.exists():
            st.image(str(p_cm_s), caption="Color + HOG + SVM (Thành viên 1)", width="stretch")


def render_tab_architecture():
    """Tab 3: Model architecture diagrams verified against actual source code."""
    st.subheader("Kiến trúc Mô hình & Đặc trưng (Đối chiếu mã nguồn thực tế)")

    sub_dl, sub_svm = st.tabs(["Enhanced LeNet-5 (Học sâu)", "Color + HOG + SVM (Cổ điển)"])

    with sub_dl:
        st.markdown("#### 1. Sơ đồ kiến trúc Enhanced LeNet-5")
        lenet_diag = PROJECT_ROOT / "assets" / "lenet_architecture.png"
        if not lenet_diag.exists():
            lenet_diag = PROJECT_ROOT / "assets" / "f3_01_lenet_architecture.png"
        if lenet_diag.exists():
            st.image(str(lenet_diag), caption="Kiến trúc thực tế từ src/detection/deep_learning/lenet_model.py (557.700 tham số)", width="stretch")

        st.markdown("#### 2. Trực quan hóa Feature Maps qua các tầng tích chập")
        fmap_path = PROJECT_ROOT / "assets" / "f3_04_lenet_feature_maps.png"
        if fmap_path.exists():
            st.image(str(fmap_path), caption="Bản đồ kích hoạt tại ConvBlock 1 và ConvBlock 3", width="stretch")

        st.markdown("#### 3. Thông số kiến trúc xác minh từ mã nguồn & checkpoint")
        i1, i2, i3 = st.columns(3)
        with i1:
            with st.container(border=True):
                st.markdown("**Đầu vào RGB 64 × 64**")
                st.caption("Resize ảnh ROI về 64×64 RGB (chuẩn hóa ImageNet), giữ rõ hoa văn biểu tượng bên trong biển báo.")
        with i2:
            with st.container(border=True):
                st.markdown("**3 ConvBlocks + Flatten 4.096**")
                st.caption("Conv 5×5 (16), Conv 5×5 (32), Conv 3×3 (64) kèm BatchNorm2d, ReLU, MaxPool 2×2; nối qua `nn.Flatten()` 4.096 → Linear(128) → Linear(4) với Dropout 0.3 & 0.2.")
        with i3:
            with st.container(border=True):
                st.markdown("**557.700 tham số (~0.56M)**")
                st.caption("Tổng tham số đếm trực tiếp từ `best_cnn.pth` (2.24 MB): 32.768 ở 3 khối Conv + 524.932 ở Classifier Head.")

    with sub_svm:
        st.markdown("#### 1. Sơ đồ pipeline trích xuất đặc trưng & phân loại S3 (457 chiều)")
        hog_pipe = PROJECT_ROOT / "assets" / "hog_svm_pipeline.png"
        if hog_pipe.exists():
            st.image(str(hog_pipe), caption="Pipeline Color + HOG + SVM từ src/detection/color_hog_svm/features.py & detector.py", width="stretch")

        st.markdown("#### 2. Trực quan hóa HOG theo nhóm biển báo")
        hog_vis = PROJECT_ROOT / "assets" / "f1_12_hog_visualization_groups.png"
        if hog_vis.exists():
            st.image(str(hog_vis), caption="Biểu diễn gradient HOG (64×64, 9 hướng, ô 16×16, block 2×2 = 324 chiều)", width="stretch")

        st.markdown("#### 3. Cấu trúc tích lũy số chiều vector đặc trưng (S0 → S3)")
        dims_path = PROJECT_ROOT / "assets" / "hog_feature_dims.png"
        if not dims_path.exists():
            dims_path = PROJECT_ROOT / "assets" / "f1_14_feature_vector_structure.png"
        if dims_path.exists():
            st.image(str(dims_path), caption="Tổng hợp số chiều qua từng nấc đặc trưng S0 (44d) → S1 (45d) → S2_2×2 (133d) → S3 (457d)", width="stretch")


def render_tab_eda():
    """Tab 4: Exploratory Data Analysis figures."""
    st.subheader("Khám phá Dữ liệu (EDA)")
    st.markdown(
        "Dataset gốc gồm **3.216 ảnh đường phố Việt Nam** với **8.334 bounding boxes** thô "
        "(8.296 boxes sau khi bỏ lớp Camera, 8.106 boxes thuộc 4 nhóm chính QCVN 41:2019)."
    )

    eda_dir = PROJECT_ROOT / "assets" / "eda"
    p1 = eda_dir / "1_class_distribution.png"
    p2 = eda_dir / "2_bbox_dimensions.png"
    p3 = eda_dir / "3_sample_traffic_signs_grid.png"
    p4 = eda_dir / "4_split_distribution.png"

    c1, c2 = st.columns(2)
    with c1:
        if p1.exists():
            st.image(str(p1), caption="Phân bố bounding boxes theo 4 nhóm mục tiêu", width="stretch")
        if p3.exists():
            st.image(str(p3), caption="Lưới ảnh mẫu 4 nhóm biển báo thực tế", width="stretch")
    with c2:
        if p2.exists():
            st.image(str(p2), caption="Phân bố kích thước & tỉ lệ khung hình (Aspect Ratio)", width="stretch")
        if p4.exists():
            st.image(
                str(p4),
                caption="Biểu đồ minh họa split cũ (70/15/15); bộ split chuẩn hóa hiện tại dùng Block Split 70/10/20 (5.254 / 542 / 1.813 mẫu)",
                width="stretch",
            )


def main():
    st.markdown(get_base_css(), unsafe_allow_html=True)

    logo_path = PROJECT_ROOT / "assets" / "logo_uit.png"
    st.markdown(render_header_html(logo_path), unsafe_allow_html=True)

    det_lenet = get_cached_lenet()
    det_svm = get_cached_svm()
    yolo_model = get_cached_yolo()

    with st.sidebar:
        st.markdown("### :material/tune: Bảng điều khiển")
        mode_choice = st.radio(
            "Chế độ kiểm thử:",
            [
                "Phân tích ảnh bất kỳ (YOLO + Phân loại)",
                "Phân loại ảnh biển đã cắt (ROI)",
            ],
            index=0,
            key="sidebar_mode",
        )
        input_mode = "scene" if "bất kỳ" in mode_choice else "roi"

        model_key = st.selectbox(
            "Bộ phân loại:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda k: MODEL_OPTIONS[k],
            index=0,
            key="sidebar_model",
        )

        if input_mode == "scene":
            st.markdown("##### :material/center_focus_weak: Tham số YOLO11s")
            yolo_conf = st.slider("Ngưỡng tin cậy (conf)", 0.10, 0.90, 0.25, 0.05, key="yolo_conf")
            yolo_iou = st.slider("Ngưỡng NMS (iou)", 0.30, 0.80, 0.60, 0.05, key="yolo_iou")
            st.caption("Kích thước suy luận cố định: `imgsz = 800`")
        else:
            yolo_conf, yolo_iou = 0.25, 0.60

        st.divider()
        st.markdown("##### :material/verified: Trạng thái Checkpoints")
        st.markdown(
            render_checkpoint_rows_html(
                [
                    ("Enhanced LeNet-5", det_lenet is not None),
                    ("Color + HOG + SVM", det_svm is not None),
                    ("YOLO11s Detector", yolo_model is not None),
                ]
            ),
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(
            "**Phân công nhóm (CS231):**\n"
            "- **Thành viên 1:** YOLO11s Detector + Color/HOG/SVM (457d)\n"
            "- **Thành viên 2:** Enhanced LeNet-5 (PyTorch CNN)\n"
            "- **GVHD:** ThS. Huỳnh Hồ Thị Mộng Trinh"
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            ":material/center_focus_strong: Thử nghiệm",
            ":material/leaderboard: Bảng xếp hạng",
            ":material/account_tree: Kiến trúc & Đặc trưng",
            ":material/analytics: Dữ liệu (EDA)",
        ]
    )

    with tab1:
        render_tab_interactive(
            det_lenet=det_lenet,
            det_svm=det_svm,
            yolo_model=yolo_model,
            input_mode=input_mode,
            model_key=model_key,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
        )
    with tab2:
        render_tab_benchmarks()
    with tab3:
        render_tab_architecture()
    with tab4:
        render_tab_eda()


if __name__ == "__main__":
    main()
