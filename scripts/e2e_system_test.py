"""
FallGuard AI - End-to-End System & API Verification Test
Tests active FastAPI endpoints, WebSocket telemetry broadcast, Simulation controls,
Fall incident registration, and Caregiver acknowledgment.
"""

import sys
import json
import asyncio
import requests
import websockets
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/monitor"

def test_rest_api_flows():
    print("\n" + "=" * 70)
    print("      FALLGUARD AI - END-TO-END SYSTEM INTEGRATION TEST")
    print("=" * 70)
    
    # 1. Health check
    print("\n[Step 1] Checking Health Endpoint (GET /api/health)...")
    res = requests.get(f"{BASE_URL}/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print(f"  -> Health Response: {res.json()}")
    
    # 2. Model Info
    print("\n[Step 2] Checking Model Info & Metadata (GET /api/model/info)...")
    res = requests.get(f"{BASE_URL}/api/model/info")
    assert res.status_code == 200, f"Model info failed: {res.text}"
    model_info = res.json()
    print(f"  -> Model Name:       {model_info['model_name']}")
    print(f"  -> Features Count:   {model_info['features_count']}")
    print(f"  -> Fall Recall:      {model_info['metrics']['fall_recall']*100:.2f}%")
    print(f"  -> Top Feature:      {model_info['feature_importance_top10'][0]['feature']}")
    
    # 3. Model Benchmark Comparisons
    print("\n[Step 3] Checking 5-Model Benchmark Comparisons (GET /api/model/comparisons)...")
    res = requests.get(f"{BASE_URL}/api/model/comparisons")
    assert res.status_code == 200
    comparisons = res.json()
    print(f"  -> Total Benchmarked Models: {len(comparisons)}")
    for m in comparisons:
        print(f"     - {m['Model']:<24}: Acc={m['Accuracy']*100:.1f}%, FallRecall={m['Fall Recall']*100:.1f}%, Score={m['Composite Score']*100:.1f}%")
        
    # 4. Single Window Predict Endpoint (Normal Walking)
    print("\n[Step 4] Testing Predict Endpoint on Normal Posture Window (POST /api/predict)...")
    walk_payload = {
        "acc_x": [0.15] * 128,
        "acc_y": [0.25] * 128,
        "acc_z": [0.98] * 128,
        "gyro_x": [0.05] * 128,
        "gyro_y": [0.05] * 128,
        "gyro_z": [0.02] * 128,
        "device_id": "ROOM_104_BED_01"
    }
    res = requests.post(f"{BASE_URL}/api/predict", json=walk_payload)
    assert res.status_code == 200
    pred_res = res.json()
    print(f"  -> Prediction: {pred_res['activity']}, Confidence: {pred_res['confidence']*100:.1f}%, Risk: {pred_res['risk_level']}")
    
    # 5. Single Window Predict Endpoint (High Impact Fall)
    print("\n[Step 5] Testing Predict Endpoint on Severe Fall Shock Signature...")
    fall_payload = {
        "acc_x": [3.8] * 128,
        "acc_y": [2.9] * 128,
        "acc_z": [-4.2] * 128,
        "gyro_x": [4.5] * 128,
        "gyro_y": [3.9] * 128,
        "gyro_z": [2.8] * 128,
        "device_id": "ROOM_104_BED_01"
    }
    res = requests.post(f"{BASE_URL}/api/predict", json=fall_payload)
    assert res.status_code == 200
    fall_pred = res.json()
    print(f"  -> Prediction: {fall_pred['activity']}, IsFall: {fall_pred['is_fall']}, Risk: {fall_pred['risk_level']}")
    assert fall_pred["is_fall"] is True
    assert fall_pred["risk_level"] == "HIGH"
    
    # 6. Fall Incidents Retrieval
    print("\n[Step 6] Verifying Fall Event Storage in Database (GET /api/falls)...")
    res = requests.get(f"{BASE_URL}/api/falls")
    assert res.status_code == 200
    falls = res.json()
    assert len(falls) > 0
    latest_fall = falls[0]
    print(f"  -> Latest Fall ID #{latest_fall['id']}, Status: {latest_fall['status']}, Device: {latest_fall['device_id']}")
    
    # 7. Caregiver Acknowledgment Flow
    print(f"\n[Step 7] Testing Caregiver Acknowledgment for Fall #{latest_fall['id']} (POST /api/falls/{latest_fall['id']}/acknowledge)...")
    ack_res = requests.post(
        f"{BASE_URL}/api/falls/{latest_fall['id']}/acknowledge",
        json={"acknowledged_by": "Senior Caregiver Sarah", "notes": "Attended within 30 seconds. Patient vitals stable."}
    )
    assert ack_res.status_code == 200
    ack_data = ack_res.json()
    assert ack_data["acknowledged"] is True
    assert ack_data["status"] == "ACKNOWLEDGED"
    print(f"  -> Acknowledged By: {ack_data['acknowledged_by']}")
    print(f"  -> Notes: {ack_data['notes']}")
    
    # 8. Dashboard Summary
    print("\n[Step 8] Checking Dashboard Aggregated Summary (GET /api/dashboard/summary)...")
    res = requests.get(f"{BASE_URL}/api/dashboard/summary")
    assert res.status_code == 200
    summary = res.json()
    print(f"  -> Total Falls Count: {summary['total_falls_count']}")
    print(f"  -> Falls Today:       {summary['falls_today_count']}")
    print(f"  -> System Status:     {summary['system_status']}")

async def test_websocket_and_simulation():
    print("\n[Step 9] Testing WebSocket Telemetry & Simulator Controls...")
    
    # Start Simulation via REST
    res = requests.post(f"{BASE_URL}/api/simulation/control", json={"action": "start", "playback_speed": 2.0})
    assert res.status_code == 200
    print("  -> Simulator Started at 2.0x speed.")
    
    # Connect WebSocket client and receive real-time ticks
    async with websockets.connect(WS_URL) as ws:
        print("  -> WebSocket connected. Listening for telemetry ticks...")
        received_ticks = 0
        for _ in range(10):
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(msg_raw)
            if msg.get("type") == "SENSOR_TICK":
                received_ticks += 1
                tick = msg["data"]
                print(f"     [Tick #{received_ticks}] Acc=({tick['acc_x']:+.2f}, {tick['acc_y']:+.2f}, {tick['acc_z']:+.2f}) Mag={tick['acc_magnitude']:.2f}g GroundTruth={tick['ground_truth']}")
                
        assert received_ticks >= 5, "Failed to receive live sensor ticks over WebSocket"
        print(f"  -> Successfully verified {received_ticks} real-time sensor ticks over WebSocket!")
        
    # Stop simulation
    requests.post(f"{BASE_URL}/api/simulation/control", json={"action": "stop"})
    print("  -> Simulator Stopped.")
    print("\n" + "=" * 70)
    print("  ALL END-TO-END INTEGRATION FLOWS VERIFIED SUCCESSFULLY!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_rest_api_flows()
    asyncio.run(test_websocket_and_simulation())
