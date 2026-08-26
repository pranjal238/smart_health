"""
FallGuard AI - Standalone Model Training CLI
Executes dataset verification, feature engineering, 5-model benchmarking,
model selection, figure generation, and metadata serialization.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.download_datasets import build_dataset_pipeline
from ml.training.trainer import train_and_evaluate_all_models

def main():
    print("\n" + "=" * 75)
    print("      FALLGUARD AI - SENSOR MODEL TRAINING & EVALUATION CLI")
    print("=" * 75 + "\n")
    
    # 1. Ensure dataset exists
    processed_csv = BASE_DIR / "data" / "processed" / "unified_sensor_stream.csv"
    if not processed_csv.exists():
        print("[1/3] Ingesting and formatting raw sensor streams...")
        build_dataset_pipeline()
    else:
        print(f"[1/3] Using compiled sensor dataset: {processed_csv.name}")
        
    # 2. Train and benchmark models
    print("\n[2/3] Extracting sliding window features and training 5 candidate classifiers...")
    bundle = train_and_evaluate_all_models(dataset_csv_path=processed_csv, sampling_rate=50.0)
    
    # 3. Print Final Report
    metrics = bundle["metrics"]
    print("\n" + "=" * 75)
    print("                      MODEL TRAINING COMPLETE")
    print("=" * 75)
    print(f"  * Best Model Architecture:  {bundle['model_name']}")
    print(f"  * Overall Accuracy:         {metrics['accuracy'] * 100:.2f}%")
    print(f"  * Macro Precision:          {metrics['macro_precision'] * 100:.2f}%")
    print(f"  * Macro Recall:             {metrics['macro_recall'] * 100:.2f}%")
    print(f"  * Macro F1-Score:           {metrics['macro_f1'] * 100:.2f}%")
    print(f"  * Fall Class Recall:        {metrics['fall_recall'] * 100:.2f}% (CRITICAL SAFETY METRIC)")
    print(f"  * Fall Class Precision:     {metrics['fall_precision'] * 100:.2f}%")
    print(f"  * Fall False Negative Rate: {metrics['fall_fnr'] * 100:.2f}%")
    print(f"  * Model Saved Location:     data/models/fall_detection_model.joblib")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
