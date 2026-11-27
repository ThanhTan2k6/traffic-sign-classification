"""CSS styling and HTML helpers for a consistent dark dashboard theme."""

import base64
from pathlib import Path
from demo_ui.constants import CLASS_METADATA


def get_base_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

/* Header Card */
.app-header {
    display: flex;
    align-items: center;
    gap: 18px;
    background: linear-gradient(135deg, #0F172A 0%, #172554 55%, #0F172A 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 18px;
}
.app-header-logo {
    flex: 0 0 auto;
    width: 64px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.94);
    border-radius: 10px;
    padding: 6px;
}
.app-header-logo img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
.app-header-body {
    flex: 1 1 auto;
    min-width: 0;
}
.app-header-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin-bottom: 4px;
}
.app-header-sub {
    font-size: 0.875rem;
    color: #CBD5E1;
    line-height: 1.45;
    margin-bottom: 10px;
}
.app-header-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.app-tag {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.12);
    color: #BAE6FD;
    white-space: nowrap;
}
.app-tag-green {
    border-color: rgba(16, 185, 129, 0.35);
    background: rgba(16, 185, 129, 0.12);
    color: #A7F3D0;
}
.app-tag-amber {
    border-color: rgba(245, 158, 11, 0.35);
    background: rgba(245, 158, 11, 0.12);
    color: #FDE68A;
}

@media (max-width: 680px) {
    .app-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
        padding: 14px 16px;
    }
    .app-header-title {
        font-size: 1.15rem;
    }
}

/* Sidebar Checkpoint Rows */
.ckpt-list {
    margin-top: 6px;
    margin-bottom: 6px;
}
.ckpt-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    font-size: 0.84rem;
}
.ckpt-row:last-child {
    border-bottom: none;
}
.ckpt-name {
    color: #CBD5E1;
    font-weight: 500;
    white-space: nowrap;
}
.ckpt-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
}
.ckpt-ok {
    background: rgba(16, 185, 129, 0.16);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.ckpt-missing {
    background: rgba(239, 68, 68, 0.16);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

/* QCVN Group Badge */
.qcvn-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.2;
}

/* Crop Card Caption */
.crop-meta {
    font-size: 0.80rem;
    color: #94A3B8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 14px;
    border-radius: 8px 8px 0 0;
}

/* Metric border adjustment */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 10px;
    padding: 12px 14px;
}
</style>
"""


def render_header_html(logo_path: Path | None) -> str:
    logo_html = "<span style='font-weight:800; color:#0F172A; font-size:1.1rem;'>UIT</span>"
    if logo_path is not None and logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_html = f"<img src='data:image/png;base64,{b64}' alt='Logo UIT' />"

    return f"""
<div class="app-header">
  <div class="app-header-logo">{logo_html}</div>
  <div class="app-header-body">
    <div class="app-header-title">HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG VIỆT NAM</div>
    <div class="app-header-sub">
      Đồ án Thị giác Máy tính (CS231) · Trường ĐH Công nghệ Thông tin (UIT – ĐHQG-HCM) · Quy chuẩn <b>QCVN 41:2019/BGTVT</b>
    </div>
    <div class="app-header-tags">
      <span class="app-tag">4 nhóm biển QCVN 41</span>
      <span class="app-tag app-tag-green">Enhanced LeNet-5 · 557.700 tham số</span>
      <span class="app-tag app-tag-amber">Color + HOG + SVM · 457 chiều</span>
      <span class="app-tag">Two-Stage YOLO11s (imgsz=800)</span>
    </div>
  </div>
</div>
"""


def render_checkpoint_rows_html(rows: list[tuple[str, bool]]) -> str:
    items = []
    for label, ok in rows:
        cls = "ckpt-ok" if ok else "ckpt-missing"
        txt = "Sẵn sàng" if ok else "Thiếu file"
        items.append(
            f"<div class='ckpt-row'>"
            f"<span class='ckpt-name'>{label}</span>"
            f"<span class='ckpt-badge {cls}'>{txt}</span>"
            f"</div>"
        )
    return f"<div class='ckpt-list'>{''.join(items)}</div>"


def render_qcvn_pill_html(class_name: str, extra_text: str = "") -> str:
    meta = CLASS_METADATA.get(class_name, CLASS_METADATA["Unknown"])
    color = meta["color_hex"]
    r, g, b = meta["color_rgb"]
    label = f"{meta['vn_name']} ({meta['code_prefix']})"
    if extra_text:
        label = f"{label} · {extra_text}"
    return (
        f"<span class='qcvn-pill' style='background:rgba({r},{g},{b},0.16); "
        f"color:{color}; border:1px solid rgba({r},{g},{b},0.45);'>{label}</span>"
    )
