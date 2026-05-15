"""
TrafficPulse — Intelligent Traffic Flow Analyzer
Streamlit dashboard powered by YOLOv8.
"""

import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from pathlib import Path
import tempfile

from detector import TrafficDetector, VEHICLE_CLASSES
from tracker import CentroidTracker, LineCrossingCounter
from analytics import HeatmapAccumulator, TrafficTimeline, congestion_level

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficPulse",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; }
    .congestion-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stSidebar { background-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/TrafficPulse-v1.0-blue?style=for-the-badge&logo=opencv")
    st.markdown("### ⚙️ Configuration")

    model_size = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                              help="Nano=fastest, Medium=most accurate")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    show_heatmap = st.toggle("Show Heatmap Overlay", value=True)
    show_counter_line = st.toggle("Show Counter Line", value=True)
    show_conf_labels = st.toggle("Show Confidence Labels", value=True)
    line_position = st.slider("Counter Line Position (%)", 20, 80, 50)

    st.markdown("---")
    st.markdown("### 📊 Filter Classes")
    selected_classes = {}
    for cls_id, (label, color) in VEHICLE_CLASSES.items():
        selected_classes[cls_id] = st.checkbox(f"{label.capitalize()}", value=True)

    st.markdown("---")
    st.markdown("**TrafficPulse** · YOLOv8 + OpenCV  \nBuilt by [DamiaYuhanes](https://github.com/DamiaYuhanes)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.8rem; margin-bottom:0;'>
    🚦 TrafficPulse
</h1>
<p style='text-align:center; color:#94a3b8; font-size:1.1rem; margin-top:4px;'>
    Real-time Intelligent Traffic Flow Analyzer · Powered by YOLOv8
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Source selection ──────────────────────────────────────────────────────────
source_tab, demo_tab = st.tabs(["📹 Upload Video", "🎬 Demo Mode"])

video_source = None
with source_tab:
    uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        video_source = tmp.name
        st.success(f"Loaded: **{uploaded.name}**")

with demo_tab:
    st.info("📌 No video? Use a webcam or download a sample traffic video from [Pexels](https://www.pexels.com/search/videos/traffic/) and upload it.")
    if st.button("▶ Use Webcam (Live)"):
        video_source = 0

# ── Main analysis ─────────────────────────────────────────────────────────────
if video_source is not None:
    start_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

    if start_btn:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Could not open video source.")
            st.stop()

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        line_y = int(H * line_position / 100)

        detector = TrafficDetector(model_size, conf_thresh)
        tracker = CentroidTracker()
        counter = LineCrossingCounter(line_y)
        heatmap = HeatmapAccumulator(H, W)
        timeline = TrafficTimeline(window=int(fps * 60))

        # Layout
        col_vid, col_stats = st.columns([3, 1])
        with col_vid:
            frame_placeholder = st.empty()
        with col_stats:
            st.markdown("#### 📈 Live Stats")
            total_box = st.empty()
            congestion_box = st.empty()
            class_chart = st.empty()

        st.divider()
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            timeline_chart = st.empty()
        with col_t2:
            crossing_chart = st.empty()

        stop_btn = st.button("⏹ Stop", use_container_width=True)

        frame_count = 0
        class_frame_counts: dict[str, int] = defaultdict(int)
        start_time = time.time()

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            frame_count += 1

            # Filter by selected classes
            detections = detector.detect(frame)
            detections = [d for d in detections if selected_classes.get(d["class_id"], True)]

            tracked = tracker.update(detections)
            counter.update(tracked, tracker)
            heatmap.update(detections)
            timeline.record(len(detections), elapsed)

            for d in detections:
                class_frame_counts[d["label"]] += 1

            # Annotate
            display = heatmap.render(frame) if show_heatmap else frame.copy()
            display = detector.annotate(display, detections, show_conf=show_conf_labels)

            if show_counter_line:
                cv2.line(display, (0, line_y), (W, line_y), (0, 255, 255), 2)
                cv2.putText(display, f"COUNT LINE | crossed: {counter.total}",
                            (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(display, f"Frame: {frame_count} | FPS: {fps:.0f} | Vehicles: {len(detections)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, use_container_width=True)

            # Stats sidebar
            level, lvl_color = congestion_level(len(detections))
            total_box.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#3B82F6'>{len(detections)}</div>
                <div class='metric-label'>Vehicles in Frame</div>
            </div>
            """, unsafe_allow_html=True)
            congestion_box.markdown(f"""
            <div class='metric-card' style='margin-top:8px'>
                <div class='metric-label'>Congestion Level</div>
                <div style='margin-top:6px'>
                    <span class='congestion-badge' style='background:{lvl_color}22; color:{lvl_color}; border:1px solid {lvl_color}'>{level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Class distribution pie
            if class_frame_counts:
                fig_pie = px.pie(
                    names=list(class_frame_counts.keys()),
                    values=list(class_frame_counts.values()),
                    color_discrete_sequence=["#3B82F6","#F59E0B","#EF4444","#8B5CF6","#10B981","#F97316"],
                    hole=0.5,
                )
                fig_pie.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    font_color="#e2e8f0", margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=True, legend=dict(font=dict(size=10)),
                    height=220,
                )
                class_chart.plotly_chart(fig_pie, use_container_width=True)

            # Timeline chart
            if frame_count % 5 == 0:
                ts, counts = timeline.as_lists()
                fig_line = go.Figure(go.Scatter(
                    x=ts, y=counts, fill="tozeroy",
                    line=dict(color="#3B82F6", width=2),
                    fillcolor="rgba(59,130,246,0.15)",
                ))
                fig_line.update_layout(
                    title="Vehicle Count Over Time",
                    xaxis_title="Seconds", yaxis_title="Count",
                    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font_color="#e2e8f0", height=260,
                    margin=dict(t=36, b=30, l=40, r=10),
                )
                timeline_chart.plotly_chart(fig_line, use_container_width=True)

                # Crossing bar chart
                if counter.counts:
                    fig_bar = px.bar(
                        x=list(counter.counts.keys()),
                        y=list(counter.counts.values()),
                        title="Line Crossing Counts",
                        color=list(counter.counts.keys()),
                        color_discrete_sequence=["#3B82F6","#F59E0B","#EF4444","#8B5CF6","#10B981","#F97316"],
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                        font_color="#e2e8f0", height=260, showlegend=False,
                        margin=dict(t=36, b=30, l=40, r=10),
                    )
                    crossing_chart.plotly_chart(fig_bar, use_container_width=True)

        cap.release()
        st.success(f"✅ Analysis complete! Processed **{frame_count}** frames · Total crossings: **{counter.total}**")

else:
    st.markdown("""
    <div style='text-align:center; padding:60px 0; color:#64748b;'>
        <div style='font-size:4rem;'>🚦</div>
        <div style='font-size:1.3rem; margin-top:12px;'>Upload a traffic video or start webcam to begin analysis</div>
        <div style='font-size:0.9rem; margin-top:8px;'>Supports MP4, AVI, MOV, MKV</div>
    </div>
    """, unsafe_allow_html=True)
