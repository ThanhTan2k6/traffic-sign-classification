"""Plotly chart builders with clean responsive margins and zero label collisions."""

from typing import Dict
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.config import TARGET_CLASSES
from demo_ui.constants import CLASS_METADATA

MODEL_STYLE_MAP = {
    "Enhanced LeNet-5 (Đề xuất)": {"color": "#38BDF8", "symbol": "star", "size": 15},
    "Color + HOG + SVM (S3)": {"color": "#F59E0B", "symbol": "diamond", "size": 12},
    "Classical LeNet-5 (Baseline)": {"color": "#94A3B8", "symbol": "square", "size": 11},
    "MobileNetV3-Small": {"color": "#10B981", "symbol": "triangle-up", "size": 12},
    "ResNet-18": {"color": "#A78BFA", "symbol": "circle", "size": 12},
}


def build_probability_bar_chart(prob_dict: Dict[str, float], title: str):
    """Build horizontal 4-class bar chart using consistent QCVN colors."""
    if not HAS_PLOTLY:
        return None

    labels = []
    values = []
    colors = []
    for cls in TARGET_CLASSES:
        meta = CLASS_METADATA.get(cls, CLASS_METADATA["Unknown"])
        labels.append(f"{meta['vn_name']} ({meta['code_prefix']})")
        values.append(prob_dict.get(cls, 0.0) * 100.0)
        colors.append(meta["color_hex"])

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.25)", width=1),
                cornerradius=5,
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=12, color="#E2E8F0", family="JetBrains Mono, monospace"),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13.5, color="#E2E8F0")),
        xaxis=dict(
            range=[0, 120],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.14)",
            ticksuffix="%",
            tickfont=dict(size=11, color="#94A3B8"),
            title="",
        ),
        yaxis=dict(
            autorange="reversed",
            automargin=True,
            tickfont=dict(size=11.5, color="#E2E8F0"),
        ),
        margin=dict(l=8, r=28, t=38, b=10),
        height=225,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_tradeoff_scatter(df_roi: pd.DataFrame):
    """
    Accuracy vs. Inference Latency scatter plot with distinct colors & symbols,
    zero overlapping annotations, X starting at 0, and explicit Y>=95% axis note.
    """
    if not HAS_PLOTLY:
        return None

    fig = go.Figure()
    for _, row in df_roi.iterrows():
        name = str(row["Mô Hình Phân Loại"])
        style = MODEL_STYLE_MAP.get(name, {"color": "#CBD5E1", "symbol": "circle", "size": 12})
        fig.add_trace(
            go.Scatter(
                x=[row["Độ Trễ (ms)"]],
                y=[row["Độ Chính Xác (%)"]],
                mode="markers",
                name=name,
                marker=dict(
                    color=style["color"],
                    symbol=style["symbol"],
                    size=style["size"],
                    line=dict(color="#0F172A", width=1.2),
                ),
                customdata=[[
                    row["Loại Mô Hình"],
                    row["Macro F1 (%)"],
                    row["Tốc Độ (FPS)"],
                    row["Số Tham Số"],
                    row["Dung Lượng (MB)"],
                ]],
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Loại: %{customdata[0]}<br>"
                    "Accuracy: %{y:.2f}%<br>"
                    "Macro F1: %{customdata[1]:.2f}%<br>"
                    "Độ trễ: %{x:.2f} ms (%{customdata[2]:.1f} FPS)<br>"
                    "Tham số/Chiều: %{customdata[3]} · Dung lượng: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Đánh Đổi Hiệu Năng: Độ Chính Xác vs Độ Trễ Suy Luận",
            font=dict(size=14, color="#F1F5F9"),
        ),
        xaxis=dict(
            title=dict(text="Độ trễ suy luận (ms) — càng thấp càng tốt", font=dict(size=12, color="#CBD5E1")),
            range=[0.0, 4.75],
            dtick=0.5,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.14)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.3)",
            tickfont=dict(size=11, color="#94A3B8"),
        ),
        yaxis=dict(
            title=dict(text="Độ chính xác (%) — trục Y bắt đầu từ 95%", font=dict(size=12, color="#CBD5E1")),
            range=[95.0, 99.6],
            dtick=1.0,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.14)",
            tickfont=dict(size=11, color="#94A3B8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#E2E8F0"),
            bgcolor="rgba(15, 23, 42, 0.45)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
        ),
        margin=dict(l=64, r=24, t=48, b=115),
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
    )
    return fig


def build_fps_bar_chart(df_roi: pd.DataFrame):
    """Sorted horizontal bar chart comparing model throughput (FPS)."""
    if not HAS_PLOTLY:
        return None

    df_sorted = df_roi.sort_values(by="Tốc Độ (FPS)", ascending=True).reset_index(drop=True)
    names = df_sorted["Mô Hình Phân Loại"].tolist()
    fps_vals = df_sorted["Tốc Độ (FPS)"].tolist()
    lat_vals = df_sorted["Độ Trễ (ms)"].tolist()
    colors = [MODEL_STYLE_MAP.get(n, {}).get("color", "#38BDF8") for n in names]
    max_fps = max(fps_vals) if fps_vals else 1200.0

    fig = go.Figure(
        go.Bar(
            x=fps_vals,
            y=names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                cornerradius=5,
            ),
            text=[f" {f:.1f} FPS ({l:.2f} ms)" for f, l in zip(fps_vals, lat_vals)],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11.5, color="#E2E8F0", family="JetBrains Mono, monospace"),
            hovertemplate="<b>%{y}</b><br>Tốc độ: %{x:.1f} FPS<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="So Sánh Tốc Độ Suy Luận (FPS — Tăng dần từ dưới lên)",
            font=dict(size=14, color="#F1F5F9"),
        ),
        xaxis=dict(
            title=dict(text="Thông lượng (FPS) — càng cao càng tốt", font=dict(size=12, color="#CBD5E1")),
            range=[0, max_fps * 1.42],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.14)",
            tickfont=dict(size=11, color="#94A3B8"),
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=11.5, color="#E2E8F0"),
        ),
        margin=dict(l=12, r=40, t=48, b=46),
        height=310,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
    )
    return fig
