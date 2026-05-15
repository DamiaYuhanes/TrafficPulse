# 🚦 TrafficPulse — Intelligent Traffic Flow Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF4500?style=for-the-badge&logo=yolo&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-Object%20Detection-10B981?style=for-the-badge"/>
</p>

<p align="center">
  <b>Real-time multi-class vehicle detection, tracking, and traffic analytics powered by YOLOv8 and a sleek Streamlit dashboard.</b>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Multi-class Detection** | Cars, trucks, buses, motorcycles, bicycles, pedestrians |
| 🔁 **Real-time Tracking** | Centroid-based tracker maintains object identity across frames |
| 📍 **Virtual Line Counter** | Count vehicles crossing a configurable line |
| 🌡️ **Heatmap Overlay** | Visualize traffic density accumulation over time |
| 📊 **Congestion Scoring** | Free Flow → Light → Moderate → Heavy → Gridlock |
| 📈 **Live Charts** | Vehicle count timeline + class distribution pie chart |
| 🎛️ **Configurable** | Model size, confidence, class filters, line position |
| 📹 **Video & Webcam** | Upload MP4/AVI/MOV or stream from webcam |

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────┬────────────────────┐
│                             │  📈 Live Stats      │
│   🎥 Live Detection Feed    │  Vehicles: 23       │
│   [Bounding boxes + labels] │  Congestion: Heavy  │
│   [Heatmap overlay]         │  [Pie Chart]        │
│   [Counter line]            │                     │
├─────────────────────────────┴────────────────────┤
│  📊 Vehicle Count Timeline  │ 🚗 Line Crossings   │
│  [Area chart over time]     │ [Bar chart per class]│
└──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/DamiaYuhanes/TrafficPulse.git
cd TrafficPulse
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run (~6MB).

### 4. Launch the dashboard
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
TrafficPulse/
├── app.py            # Streamlit dashboard (main entry point)
├── detector.py       # YOLOv8 detection engine + annotation
├── tracker.py        # Centroid tracker + line crossing counter
├── analytics.py      # Heatmap, congestion scoring, timeline
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 🧠 How It Works

```
Video Frame
    │
    ▼
┌─────────────┐
│  YOLOv8     │  ← Detects objects (class, bbox, confidence)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Centroid   │  ← Tracks objects across frames with unique IDs
│  Tracker    │
└──────┬──────┘
       │
       ├──► Line Crossing Counter  (counts vehicles per class)
       ├──► Heatmap Accumulator    (spatial density over time)
       └──► Timeline Recorder      (count per second)
                │
                ▼
        ┌──────────────┐
        │  Streamlit   │  ← Renders annotated frames + charts
        │  Dashboard   │
        └──────────────┘
```

---

## ⚙️ Configuration Options

| Option | Default | Description |
|---|---|---|
| Model | `yolov8n.pt` | Nano (fast) / Small / Medium accuracy |
| Confidence | `0.40` | Minimum detection confidence |
| Heatmap | On | Overlay spatial density visualization |
| Counter Line | 50% | Vertical position of counting line |
| Class Filter | All | Toggle individual vehicle classes |

---

## 🧪 Tested With

- Python 3.10, 3.11, 3.12
- Windows 11 / Ubuntu 22.04
- CPU inference (real-time on YOLOv8n) and GPU (CUDA)

---

## 🛣️ Roadmap

- [ ] Speed estimation (pixels/frame → km/h calibration)
- [ ] Multi-lane tracking with lane assignment
- [ ] Export analytics report as PDF
- [ ] RTSP/IP camera stream support
- [ ] Vehicle re-identification across camera cuts

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Made with ❤️ by <a href="https://github.com/DamiaYuhanes">DamiaYuhanes</a> · Powered by <a href="https://ultralytics.com">Ultralytics YOLOv8</a></p>
