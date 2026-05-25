# 🎭🧤 Mask & Gloves Detector

An AI-powered real-time safety monitoring system that detects whether a person is wearing a **face mask** and **gloves** using a webcam. On violation, it automatically captures a photo, records a 5-second video, and sends an email alert with evidence attached.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training on Google Colab](#-training-on-google-colab)
- [Running the Detector](#-running-the-detector)
- [Email Setup](#-email-setup)
- [Model Performance](#-model-performance)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Pros & Cons](#-pros--cons)
- [Future Improvements](#-future-improvements)

---

## 🧠 Project Overview

This project is a **computer vision safety system** built using:
- **YOLOv8** (Ultralytics) — for mask and gloves detection
- **OpenCV** — for webcam feed, drawing boxes, and face zone detection
- **Python** — core language
- **Gmail SMTP** — for automated email alerts

The system monitors a live webcam feed and enforces PPE (Personal Protective Equipment) compliance by detecting:
- ✅ `with_mask` — Person is wearing a mask correctly
- ⚠️ `mask_weared_incorrect` — Mask is worn incorrectly
- ❌ `without_mask` — No mask detected
- ✅ `with_gloves` — Person is wearing gloves
- ❌ `without_gloves` — No gloves detected

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 Real-time detection | Live webcam feed with instant AI detection |
| 🎭 Mask detection | 3-class: with_mask, without_mask, mask_weared_incorrect |
| 🧤 Gloves detection | 2-class: with_gloves, without_gloves |
| 🚫 Face zone protection | Prevents forehead/face being detected as a hand |
| 📸 Auto snapshot | Saves violation image automatically |
| 🎬 5-second video | Records video clip at moment of violation |
| 📧 Email alerts | Sends email with photo + video attached |
| ⏱️ Cooldown system | 10-second gap between each detection cycle |
| 📊 Live status banner | Shows ALL CLEAR / VIOLATION / RECORDING / COOLDOWN |
| 🔴 Recording progress bar | Visual bar showing video recording progress |

---

## 📁 Project Structure

```
mask_gloves_detector/
│
├── venv/                          # Python virtual environment
│
├── datasets/
│   ├── mask_dataset/
│   │   ├── train/
│   │   │   ├── images/            # 5271 training images
│   │   │   └── labels/            # YOLO format labels
│   │   ├── valid/
│   │   │   ├── images/            # 471 validation images
│   │   │   └── labels/
│   │   └── data.yaml              # Dataset config
│   │
│   └── gloves_dataset/
│       ├── train/
│       │   ├── images/            # 336 training images
│       │   └── labels/
│       ├── valid/
│       │   ├── images/            # 84 validation images
│       │   └── labels/
│       └── data.yaml
│
├── models/
│   ├── mask_best.pt               # Trained mask detector model
│   └── gloves_best.pt             # Trained gloves detector model
│
├── captures/                      # Auto-saved violation images & videos
│   ├── violation_20240101_120000.jpg
│   └── violation_20240101_120000.mp4
│
├── runs/                          # Training & evaluation results
│   ├── mask/
│   ├── gloves/
│   └── evaluation/
│
├── prepare_datasets.py            # Dataset preparation script
├── train.py                       # Local training script
├── evaluate.py                    # Model evaluation script
├── detect.py                      # Main detection script
└── README.md                      # This file
```

---

## 🛠️ Requirements

### System Requirements
- Python 3.8 or higher
- Webcam
- Internet connection (for email)
- GPU optional (CPU works, GPU is faster)

### Python Libraries

```txt
ultralytics
opencv-python
torch
numpy
```

---

## ⚙️ Installation

### Step 1 — Clone or create the project folder

```bash
mkdir mask_gloves_detector
cd mask_gloves_detector
```

### Step 2 — Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install ultralytics opencv-python matplotlib Pillow
```

### Step 4 — Verify installation

```python
import ultralytics
import cv2
import torch

print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("GPU Available:", torch.cuda.is_available())
```

---

## 🗂️ Dataset Preparation

### Datasets Used

| Dataset | Source | Images | Classes |
|---|---|---|---|
| Face Mask Detection | Roboflow Universe | 5,742 total | 3 |
| Gloves Detection | Roboflow Universe | 420 total | 2 |

### Download Datasets

1. Go to [https://universe.roboflow.com](https://universe.roboflow.com)
2. Search **"face mask detection"** → Download in **YOLOv8 format**
3. Search **"gloves detection"** → Download in **YOLOv8 format**
4. Place both inside `datasets/` folder

### Run Preparation Script

```bash
python prepare_datasets.py
```

This script will:
- Split gloves dataset into 80% train / 20% valid (since it had no valid folder)
- Fix `data.yaml` paths for both datasets
- Verify all folders and file counts

**Expected output:**
```
✅ Train images : 336
✅ Valid  images : 84
✅ Gloves yaml saved
✅ Mask yaml saved
🎉 All folders verified! Ready for training.
```

---

## ☁️ Training on Google Colab

Training was done on **Google Colab Free GPU (T4)** for speed. CPU training takes 3–5 hours; GPU takes 25–40 minutes.

### Steps

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Upload datasets to Google Drive under `mask_gloves_detector/datasets/`
4. Run the following cells:

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Install
!pip install ultralytics -q
import torch
print("GPU:", torch.cuda.is_available())

# Cell 3 — Train Mask Model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data   = "/content/drive/MyDrive/mask_gloves_detector/datasets/mask_dataset/data.yaml",
    epochs = 50,
    imgsz  = 640,
    batch  = 16,
    device = 0,
    project= "/content/drive/MyDrive/mask_gloves_detector/runs/mask",
    name   = "mask_detector"
)

# Cell 4 — Train Gloves Model
model2 = YOLO("yolov8n.pt")
model2.train(
    data   = "/content/drive/MyDrive/mask_gloves_detector/datasets/gloves_dataset/data.yaml",
    epochs = 50,
    imgsz  = 640,
    batch  = 16,
    device = 0,
    project= "/content/drive/MyDrive/mask_gloves_detector/runs/gloves",
    name   = "gloves_detector"
)
```

5. Download `best.pt` from both `runs/mask/mask_detector/weights/` and `runs/gloves/gloves_detector/weights/`
6. Rename and place in local `models/` folder:
   - `models/mask_best.pt`
   - `models/gloves_best.pt`

---

## 🚀 Running the Detector

```bash
python detect.py
```

The webcam will open automatically. The system runs continuously until you press **Q** to quit.

---

## 📧 Email Setup

### Step 1 — Enable Gmail App Password

1. Go to [https://myaccount.google.com/security](https://myaccount.google.com/security)
2. Enable **2-Step Verification**
3. Search **"App Passwords"**
4. Generate a password for **Mail / Windows Computer**
5. Copy the 16-character password

### Step 2 — Update detect.py

```python
SENDER_EMAIL    = "your_email@gmail.com"      # Your Gmail
SENDER_PASSWORD = "xxxx xxxx xxxx xxxx"       # 16-char App Password
RECEIVER_EMAIL  = "receiver@example.com"      # Alert recipient
```

### What the email contains

- **Subject:** `Safety Violation Detected — 20240101_120000`
- **Body:** Timestamp + violation types detected
- **Attachment 1:** `violation.jpg` — snapshot of the violation
- **Attachment 2:** `violation_clip.mp4` — 5-second video clip

---

## 📊 Model Performance

### Mask Detector (YOLOv8n — trained on 5,271 images)

| Metric | Score |
|---|---|
| mAP50 | 0.727 🟡 |
| mAP50-95 | 0.484 |
| Precision | 0.756 |
| Recall | 0.720 |

| Class | mAP50 |
|---|---|
| mask_weared_incorrect | 0.913 🟢 |
| with_mask | 0.730 🟡 |
| without_mask | 0.538 🟠 |

### Gloves Detector (YOLOv8n — trained on 336 images)

| Metric | Score |
|---|---|
| mAP50 | 0.947 🟢 |
| mAP50-95 | 0.583 |
| Precision | 0.950 |
| Recall | 0.932 |

| Class | mAP50 |
|---|---|
| with_gloves | 0.915 🟢 |
| without_gloves | 0.978 🟢 |

---

## 🔄 How It Works

The system uses a **3-phase state machine**:

```
┌─────────┐   violation   ┌───────────┐  5 sec done  ┌──────────┐
│  IDLE   │ ────────────→ │ RECORDING │ ────────────→ │ COOLDOWN │
│         │               │           │               │          │
│ Watch   │               │ Record    │               │ Wait     │
│ & Detect│               │ + Email   │               │ 10 sec   │
└─────────┘               └───────────┘               └──────────┘
     ↑                                                      │
     └──────────────────────────────────────────────────────┘
                      cooldown done
```

### Detection Flow Per Frame

```
Webcam Frame
     │
     ├─→ OpenCV Face Detector → Face Zones (with 35% padding)
     │
     ├─→ Mask Model (YOLOv8) → Draw boxes → Check violation
     │
     └─→ Gloves Model (YOLOv8) → Skip face zones → Draw boxes → Check violation
              │
              ▼
         Violation?
         YES → Save photo + Start 5s video recording
         NO  → Continue watching
```

### Face Zone Protection

To prevent the gloves model from detecting the forehead as a hand, an exclusion zone is created around every detected face with 35% padding. Any gloves detection whose center point falls inside this zone is ignored.

---

## 🔧 Configuration

All settings are at the top of `detect.py`:

```python
COOLDOWN_SEC   = 10     # Seconds between detection cycles
CONF_MASK      = 0.50   # Minimum confidence for mask detection
CONF_GLOVES    = 0.60   # Minimum confidence for gloves detection
FACE_PADDING   = 0.35   # Face exclusion zone padding (0.0 to 1.0)
VIDEO_DURATION = 5      # Length of recorded video in seconds
WRITE_FPS      = 15     # Fixed FPS for video writing
```

**Tuning tips:**
- Increase `CONF_GLOVES` to `0.70` if false detections still occur
- Increase `FACE_PADDING` to `0.50` if forehead is still misdetected
- Decrease `CONF_MASK` to `0.40` if mask is not being detected far away
- Change `COOLDOWN_SEC` to any value based on your monitoring needs

---

## 👍 Pros & Cons

### ✅ Pros

- **Real-time** — Instant detection on live webcam feed
- **Dual detection** — Mask and gloves both in one system
- **Automated evidence** — Photo + video + email without human involvement
- **Face protection** — Smart face zone exclusion prevents false detections
- **CPU compatible** — Works without a GPU
- **Clear state machine** — No overlapping cycles, clean flow
- **Configurable** — Easy to tune thresholds and timings

### ❌ Cons

- **without_mask accuracy is 53%** — Limited by small dataset size
- **CPU is slow** — ~13 FPS on CPU vs ~60+ FPS on GPU
- **Small gloves dataset** — Only 420 images; more data would improve accuracy
- **Single camera** — No multi-camera support currently
- **Large email attachments** — Video file may be large on slow internet

---

## 🚀 Future Improvements

- [ ] Add multi-camera support
- [ ] Build a web dashboard (Flask/Streamlit) for live monitoring
- [ ] Add a database to log all violations with timestamps
- [ ] Train with larger datasets for better accuracy
- [ ] Add GPU support for faster real-time detection
- [ ] Add WhatsApp or SMS alerts in addition to email
- [ ] Add person tracking (assign IDs to individuals)
- [ ] Support video file input in addition to webcam

---

## 👨‍💻 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| Ultralytics YOLOv8 | 8.4.41 | AI detection models |
| OpenCV | 4.x | Camera, drawing, face detection |
| PyTorch | 2.11 | Deep learning backend |
| NumPy | latest | Array operations |
| smtplib | built-in | Email sending |
| Google Colab | — | GPU training |
| Roboflow | — | Dataset source |

---

## 📜 License

This project is for educational and research purposes.

---

*Built with ❤️ using YOLOv8 + OpenCV*
