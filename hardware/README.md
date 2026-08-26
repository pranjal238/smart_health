# FallGuard AI - Wearable IoT Hardware Integration Guide

This guide details the integration of physical wearable hardware (**ESP32 NodeMCU + MPU-6050 6-Axis Inertial Measurement Unit**) with the FallGuard AI backend.

---

## 1. Hardware Components

| Component | Specification | Purpose |
|---|---|---|
| **Microcontroller** | ESP32-WROOM-32 (240MHz Dual Core, Wi-Fi / BLE) | Sensor sampling & telemetry transmission |
| **IMU Sensor** | MPU-6050 (3-Axis Accelerometer + 3-Axis Gyroscope) | Motion kinematics & impact detection |
| **Power Source** | 3.7V 500mAh LiPo Battery + TP4056 Charging Module | Wearable waist / pendant deployment |
| **Interconnects** | 4-Pin JST / Breadboard Wires | I2C Communication Bus |

---

## 2. Wiring & Pinout

Connect the MPU-6050 breakout board to the ESP32 module using the standard hardware I2C bus:

```text
┌──────────────────────────┐          ┌──────────────────────────┐
│       ESP32 Board        │          │    MPU-6050 IMU Sensor   │
│                          │          │                          │
│                     3.3V ├──────────┤ VCC                      │
│                      GND ├──────────┤ GND                      │
│           GPIO 21 (SDA)  ├──────────┤ SDA                      │
│           GPIO 22 (SCL)  ├──────────┤ SCL                      │
└──────────────────────────┘          └──────────────────────────┘
```

> [!NOTE]
> Ensure the MPU-6050 AD0 pin is connected to GND (default I2C address `0x68`) or 3.3V (alternate address `0x69`).

---

## 3. Sensor Configuration & Calibration

- **Sampling Frequency**: $50\text{ Hz}$ ($20\text{ ms}$ interval between readings).
- **Accelerometer Range**: $\pm 8\text{ g}$ ($4096\text{ LSB/g}$). Captures severe fall impact deceleration spikes (typically $3\text{g} - 6\text{g}$) without register saturation.
- **Gyroscope Range**: $\pm 1000^\circ/\text{s}$ ($32.8\text{ LSB}/(^\circ/\text{s})$), converted to $\text{rad/s}$ in firmware before serialization.
- **Sliding Window Protocol**: 128 samples ($2.56\text{ seconds}$ duration) with a 50% sliding overlap (64 samples step = new prediction every $1.28\text{s}$).

---

## 4. Telemetry Transmission Format

The ESP32 transmits structured JSON telemetry via HTTP `POST /api/predict` or WebSocket `/ws/monitor`:

```json
{
  "device_id": "WEARABLE_ESP32_01",
  "sampling_rate": 50.0,
  "acc_x": [0.04, 0.05, ...],
  "acc_y": [0.12, 0.14, ...],
  "acc_z": [0.98, 0.97, ...],
  "gyro_x": [0.01, 0.02, ...],
  "gyro_y": [0.00, 0.01, ...],
  "gyro_z": [-0.01, 0.00, ...]
}
```

---

## 5. Wearable Placement Guidelines

For optimal fall detection and activity classification accuracy:
1. **Waist / Belt Clip (Recommended)**: Closest to the human center of mass (COM); provides the most reliable biomechanical signal during slips, trips, and postural transitions.
2. **Chest / Pendant**: Highly effective for syncope/faint falls and tilt angle measurement.
3. **Wrist**: Practical for smartwatches, but requires dynamic arm-swing filtering (supported by jerk and frequency domain features).
