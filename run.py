"""
FallGuard AI - Central Launcher
AI-Based Wearable Fall Detection and Real-Time Activity Monitoring System
"""

import os
import sys
import webbrowser
import uvicorn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.core.config import settings

def main():
    print("\n" + "=" * 70)
    print("      [FALLGUARD AI] - WEARABLE MONITORING PLATFORM")
    print("      AI-Based Fall Detection & Real-Time Activity System")
    print("=" * 70)
    print(f"  - Server URL:       http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"  - Web Dashboard:    http://{settings.API_HOST}:{settings.API_PORT}/")
    print(f"  - API Docs (Swagger): http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"  - API Docs (ReDoc):   http://{settings.API_HOST}:{settings.API_PORT}/redoc")
    print(f"  - WebSocket Stream: ws://{settings.API_HOST}:{settings.API_PORT}/ws/monitor")
    print("=" * 70)
    print("  Default Credentials:")
    print("    - Admin:     admin@fallguard.ai / admin123")
    print("    - Caregiver: caregiver@fallguard.ai / caregiver123")
    print("=" * 70 + "\n")
    
    # Check if model exists; if not, train automatically
    model_path = BASE_DIR / "data" / "models" / "fall_detection_model.joblib"
    if not model_path.exists():
        print("[INFO] Model not found. Running initial training pipeline...")
        from scripts.train_model import main as train_main
        train_main()
        
    print("[INFO] Starting FastAPI Web Server & WebSocket Engine...")
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )

if __name__ == "__main__":
    main()
