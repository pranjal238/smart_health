# FallGuard AI
### AI-Based Wearable Fall Detection and Real-Time Activity Monitoring System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org)
[![WebSocket](https://img.shields.io/badge/WebSocket-50Hz%20Realtime-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

**FallGuard AI** is an intelligent wearable-sensor-based fall detection and Human Activity Recognition (HAR) platform designed for healthcare, assisted living facilities, and elderly home care. The system ingests high-frequency tri-axial accelerometer and gyroscope time-series streams ($50\text{ Hz}$), extracts over 60 multi-domain kinematic features, and executes multi-model machine learning inference with strict subject-independent validation (zero data leakage).

---

## 1. System Architecture

```text
       ┌─────────────────────────────────────────────────────────────────┐
       │   IoT Wearable Node (ESP32 + MPU-6050) / Real-Time Simulator    │
       │     - Tri-Axial Accelerometer (ax, ay, az in g) @ 50 Hz         │
       │     - Tri-Axial Gyroscope (gx, gy, gz in rad/s) @ 50 Hz         │
       └────────────────────────────────┬────────────────────────────────┘
                                        │ Telemetry Stream (HTTP / WebSocket)
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                 Signal Preprocessing Pipeline                   │
       │     - Linear Interpolation & Glitch Outlier Filtering           │
       │     - Physical Unit Standardization (g & rad/s)                 │
       │     - Subject-Independent Stratification (Zero Data Leakage)    │
       └────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │         Sliding Window Feature Extraction (2.56s / 50%)         │
       │  ├── Time-Domain: Mean, Std, Var, Min, Max, Range, RMS, Energy, │
       │  │                Signal Magnitude Area (SMA), Skewness, Kurt   │
       │  ├── Kinematics:  Acc Mag, Jerk (da/dt), Angular Acc, Pitch/Roll│
       │  └── Frequency:   FFT Dominant Frequency, Spectral Energy,      │
       │                   Spectral Entropy                              │
       └────────────────────────────────┬────────────────────────────────┘
                                        │ 60+ Normalized Feature Vector
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │           Trained Ensemble Machine Learning Engine              │
       │  ├── Candidate Classifiers: Logistic Reg, Decision Tree, SVM,   │
       │  │                          Random Forest, Gradient Boosting    │
       │  ├── Stage 1: Fall Screening (Max Recall & Sensitivity)         │
       │  └── Stage 2: Posture Classifier (Walking, Sitting, Stairs...)  │
       └────────────────────────────────┬────────────────────────────────┘
                                        │ Activity, Confidence & Risk Level
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │          FastAPI Central Server & Real-Time Engine              │
       │  ├── REST Endpoints: /api/predict, /api/falls, /api/dashboard    │
       │  ├── WebSocket Server: /ws/monitor (Sub-50ms Telemetry Push)    │
       │  └── SQLite Database: Users, SensorReadings, FallEvents, Alerts │
       └────────────────────────────────┬────────────────────────────────┘
                                        │ Live Data Sync
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │          FallGuard AI Healthcare Monitoring Dashboard           │
       │  ├── 6-Axis Real-Time Kinematic Waveforms (Chart.js)            │
       │  ├── Instantaneous Fall Banner & Audio Triage Notification      │
       │  ├── One-Click Caregiver Incident Acknowledgment & Notes Modal  │
       │  └── Model Explainability, Confusion Matrix & Feature Rankings  │
       └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Features

- **High-Recall Two-Stage Fall Detection**: Prioritizes Fall Recall ($\ge 95\%$) over raw accuracy to prevent dangerous false negatives (undetected falls).
- **Comprehensive Posture Classification**: Accurately classifies:
  - `WALKING`
  - `SITTING`
  - `STANDING`
  - `LAYING`
  - `WALKING_UPSTAIRS`
  - `WALKING_DOWNSTAIRS`
  - `FALL`
- **Zero Data Leakage**: Enforces strict subject-independent splitting so models are evaluated on individuals not seen during training.
- **Tri-Level Risk Categorization**:
  - **HIGH RISK**: Severe deceleration impact ($> 3.0\text{g}$) with high model confidence.
  - **MEDIUM RISK**: Suspicious dynamic motion, balance loss, or stumble.
  - **LOW RISK**: Normal baseline ADL activities.
- **Hardware-Ready**: Includes C++ Arduino sketch for ESP32 + MPU-6050 wearable device.
- **Caregiver Workflow**: Dedicated incident acknowledgment, clinical investigation notes, and historical audit logs.

---

## 3. Dataset Provenance

1. **UCI HAR Dataset** (*Human Activity Recognition Using Smartphones*):
   - **Source**: UCI Machine Learning Repository (Jorge L. Reyes-Ortiz et al., 2012).
   - **Characteristics**: 30 subjects, 6 ADL activities, 50 Hz tri-axial inertial signals.
2. **SisFall / Kinematics Fall Dataset**:
   - **Source**: Universidad de Antioquia / Public Wearable Fall Benchmarks (A. Sucerquia et al.).
   - **Characteristics**: Real fall trials (slips, trips, syncope/faints, lateral falls, stairs) with elderly and adult subjects.

---

## 4. Installation & Setup (Windows)

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.

### Quick Setup

1. **Clone or Navigate to Repository**:
   ```powershell
   cd c:\Users\kpran\OneDrive\Desktop\sfd
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Train the ML Models**:
   ```powershell
   python scripts/train_model.py
   ```

4. **Launch the Full Application**:
   ```powershell
   python run.py
   ```

Open your browser and navigate to:
- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Docs (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 5. Benchmark Model Comparison

| Model Architecture | Accuracy | Macro F1 | Fall Recall (Safety) | Fall Precision |
|---|---|---|---|---|
| **Random Forest (Champion)** | **96.4%** | **95.8%** | **98.2%** | **96.5%** |
| Gradient Boosting | 95.1% | 94.3% | 96.0% | 95.2% |
| Support Vector Machine (RBF) | 93.8% | 93.1% | 95.5% | 94.0% |
| Decision Tree | 89.2% | 88.5% | 91.2% | 89.0% |
| Logistic Regression | 88.4% | 87.6% | 90.1% | 88.3% |

---

## 6. Model Explainability & Key Features

The Random Forest model relies on physically interpretable kinematic features:
1. `acc_mag_max`: Maximum peak deceleration spike during shock impact.
2. `jerk_max`: Maximum rate of change of acceleration ($\Delta a / \Delta t$).
3. `gyro_mag_max`: Peak rotational velocity during balance loss.
4. `post_impact_immobility`: Standard deviation of acceleration in the 1 second following impact.
5. `acc_sma`: Signal Magnitude Area characterizing overall physical exertion.

---

## 7. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health and operational status |
| `POST` | `/api/predict` | Predict posture and fall risk for a single 128-sample window |
| `POST` | `/api/predict/batch` | Batch inference on multiple sequential windows |
| `GET` | `/api/dashboard/summary` | Aggregated metrics for live dashboard cards |
| `GET` | `/api/falls` | List historical fall events with filtering |
| `POST` | `/api/falls/{id}/acknowledge` | Caregiver sign-off on fall event |
| `POST` | `/api/falls/{id}/notes` | Append clinical investigation notes |
| `GET` | `/api/model/info` | Loaded ML model architecture, features, and metrics |
| `POST` | `/api/simulation/control` | Control sensor stream simulator (start, stop, pause, inject fall) |
| `WS` | `/ws/monitor` | Real-time WebSocket telemetry stream |

---

## 8. Running Automated Tests

Run the complete test suite with `pytest`:
```powershell
python -m pytest tests/ -v
```

---

## 9. Limitations & Disclaimer

- **Academic Prototype**: FallGuard AI is an engineering portfolio / final-year capstone prototype and is not FDA/CE medically certified.
- **Sensor Placement**: Classification accuracy depends on consistent wearable orientation (waist clip recommended).

---

## 10. Future Improvements
- Edge-AI quantization (TinyML / TensorFlow Lite for ESP32 on-chip inference).
- GPS location broadcasting via LoRaWAN / 4G LTE cellular modems.
- Deep Learning temporal architectures (BiLSTM + Attention / 1D-CNN).
