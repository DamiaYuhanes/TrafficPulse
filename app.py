"""
TrafficPulse — Intelligent Traffic Flow Analyzer
Streamlit dashboard powered by YOLOv8 + ByteTrack.
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
from tracker import LineCrossingCounter
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
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2.4rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 2px; }
    .congestion-badge {
        display: inline-block;
        padding: 5px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 6px;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stSidebar { background-color: #1e293b; }
    div[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 TrafficPulse")
    st.markdown("### ⚙️ Detection Settings")

    model_size = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=1,
        help="n=fastest · s=balanced · m=accurate · l=most accurate (slow)",
    )
    imgsz = st.select_slider(
        "Inference Resolution",
        options=[320, 480, 640, 960, 1280],
        value=1280,
        help="Higher = detects smaller objects, slower",
    )
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.80, 0.25, 0.05,
                            help="Lower = more detections, may include false positives")

    st.markdown("### 🎨 Overlay Settings")
    show_heatmap   = st.toggle("Heatmap Overlay",    value=True)
    show_conf      = st.toggle("Confidence Labels",  value=True)
    show_track_ids = st.toggle("Track IDs",          value=True)
    show_line      = st.toggle("Counter Line",        value=True)
    use_enhance    = st.toggle("CLAHE Enhancement",   value=True,
                               help="Boosts contrast to detect objects in dark/flat areas")
    line_pos       = st.slider("Counter Line Position (%)", 20, 80, 50)

    st.markdown("### 📊 Class Filters")
    selected = {}
    cols = st.columns(2)
    for i, (cls_id, (label, color)) in enumerate(VEHICLE_CLASSES.items()):
        with cols[i % 2]:
            selected[cls_id] = st.checkbox(label.capitalize(), value=True)

    st.markdown("---")
    st.caption("TrafficPulse · YOLOv8 + ByteTrack\nBy [DamiaYuhanes](https://github.com/DamiaYuhanes)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.6rem; margin-bottom:0;'>🚦 TrafficPulse</h1>
<p style='text-align:center; color:#94a3b8; font-size:1.05rem; margin-top:4px;'>
    Real-time Traffic Flow Analyzer · YOLOv8 + ByteTrack · CLAHE Enhancement
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Source ────────────────────────────────────────────────────────────────────
src_tab, demo_tab = st.tabs(["📹 Upload Video", "📷 Webcam / Info"])

video_source = None
with src_tab:
    uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        video_source = tmp.name
        st.success(f"Loaded: **{uploaded.name}**")

with demo_tab:
    st.info("Download a sample traffic video from [Pexels](https://www.pexels.com/search/videos/traffic/) and upload it, or click below for webcam.")
    if st.button("▶ Use Webcam"):
        video_source = 0

# ── Analysis ──────────────────────────────────────────────────────────────────
if video_source is not None:
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Could not open video source.")
            st.stop()

        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        line_y = int(H * line_pos / 100)

        detector = TrafficDetector(model_size, conf_thresh, imgsz, use_enhance)
        counter  = LineCrossingCounter(line_y)
        heatmap  = HeatmapAccumulator(H, W)
        timeline = TrafficTimeline(window=int(fps * 60))

        # ── Layout ────────────────────────────────────────────────────────────
        col_vid, col_stats = st.columns([3, 1])
        with col_vid:
            frame_ph = st.empty()
        with col_stats:
            st.markdown("#### 📈 Live Stats")
            total_ph     = st.empty()
            congestion_ph = st.empty()
            pie_ph        = st.empty()

        st.divider()
        col_tl, col_bar = st.columns(2)
        with col_tl:
            tl_ph = st.empty()
        with col_bar:
            bar_ph = st.empty()

        stop = st.button("⏹ Stop", use_container_width=True)

        frame_count = 0
        class_accum: dict[str, int] = defaultdict(int)
        start_time = time.time()

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            elapsed = time.time() - start_time

            # Detection + tracking
            detections = detector.track(frame)
            detections = [d for d in detections if selected.get(d["class_id"], True)]

            counter.update(detections)
            heatmap.update(detections)
            timeline.record(len(detections), elapsed)

            for d in detections:
                class_accum[d["label"]] += 1

            # Annotate
            display = heatmap.render(frame) if show_heatmap else frame.copy()
            display = detector.annotate(display, detections,
                                        show_conf=show_conf,
                                        show_track_id=show_track_ids)

            if show_line:
                cv2.line(display, (0, line_y), (W, line_y), (0, 255, 255), 2)
                cv2.putText(display,
                            f"COUNT LINE | crossed: {counter.total}",
                            (10, line_y - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            # Info bar
            info = (f"Frame {frame_count} | "
                    f"Model: {model_size} | "
                    f"Res: {imgsz}px | "
                    f"Detections: {len(detections)}")
            cv2.rectangle(display, (0, 0), (W, 34), (0, 0, 0), -1)
            cv2.putText(display, info, (8, 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            frame_ph.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                           use_container_width=True)

            # ── Stat cards ────────────────────────────────────────────────────
            level, lvl_color = congestion_level(len(detections))
            total_ph.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#3B82F6'>{len(detections)}</div>
                <div class='metric-label'>Vehicles in Frame</div>
            </div>""", unsafe_allow_html=True)

            congestion_ph.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Congestion Level</div>
                <span class='congestion-badge'
                      style='background:{lvl_color}22;color:{lvl_color};border:1px solid {lvl_color}'>
                    {level}
                </span>
            </div>""", unsafe_allow_html=True)

            # Pie chart
            if class_accum:
                labels = list(class_accum.keys())
                values = list(class_accum.values())
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.55,
                    marker_colors=["#3B82F6","#F59E0B","#EF4444","#8B5CF6","#10B981","#F97316"],
                    textinfo="label+percent",
                    textfont_size=11,
                ))
                fig_pie.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    font_color="#e2e8f0", margin=dict(t=8, b=8, l=8, r=8),
                    showlegend=False, height=200,
                )
                pie_ph.plotly_chart(fig_pie, use_container_width=True,
                                    key=f"pie_{frame_count}")

            # Charts every 5 frames
            if frame_count % 5 == 0:
                ts, cnts = timeline.as_lists()
                fig_line = go.Figure(go.Scatter(
                    x=ts, y=cnts, fill="tozeroy",
                    line=dict(color="#3B82F6", width=2),
                    fillcolor="rgba(59,130,246,0.15)",
                    name="vehicles",
                ))
                fig_line.update_layout(
                    title=dict(text="Vehicle Count Over Time", font_size=13),
                    xaxis_title="Seconds", yaxis_title="Count",
                    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font_color="#e2e8f0", height=250,
                    margin=dict(t=36, b=30, l=40, r=10),
                )
                tl_ph.plotly_chart(fig_line, use_container_width=True,
                                   key=f"tl_{frame_count}")

                if counter.counts:
                    clabels = list(counter.counts.keys())
                    cvals   = list(counter.counts.values())
                    fig_bar = go.Figure(go.Bar(
                        x=clabels, y=cvals,
                        marker_color=["#3B82F6","#F59E0B","#EF4444","#8B5CF6","#10B981","#F97316"][:len(clabels)],
                        text=cvals, textposition="outside",
                    ))
                    fig_bar.update_layout(
                        title=dict(text="Line Crossings by Class", font_size=13),
                        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                        font_color="#e2e8f0", height=250, showlegend=False,
                        margin=dict(t=36, b=30, l=40, r=10),
                    )
                    bar_ph.plotly_chart(fig_bar, use_container_width=True,
                                        key=f"bar_{frame_count}")

        cap.release()
        st.success(
            f"✅ Done! Processed **{frame_count}** frames · "
            f"Total crossings: **{counter.total}** · "
            f"Unique tracks: **{len(counter.crossed)}**"
        )
else:
    st.markdown("""
    <div style='text-align:center; padding:60px 0; color:#64748b;'>
        <div style='font-size:4rem;'>🚦</div>
        <div style='font-size:1.2rem; margin-top:12px;'>Upload a traffic video to begin</div>
        <div style='font-size:0.85rem; margin-top:6px;'>MP4 · AVI · MOV · MKV</div>
    </div>""", unsafe_allow_html=True)
