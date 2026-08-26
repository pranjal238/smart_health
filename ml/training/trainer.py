"""
FallGuard AI - Multi-Model Training, Evaluation & Selection Engine
Trains, benchmarks, and serializes 5 baseline & ensemble classifiers.
Strictly enforces subject-independent train/test isolation to prevent data leakage.
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from ml.features.feature_engineering import extract_dataset_features
from ml.preprocessing.cleaner import clean_sensor_stream
from ml.preprocessing.splitter import split_by_subject
from ml.evaluation.evaluator import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
    plot_activity_distribution,
    FIGURES_DIR,
    REPORTS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODELS_DIR / "fall_detection_model.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "model_metrics.json"
COMPARISON_JSON_PATH = REPORTS_DIR / "model_comparison_report.json"

def train_and_evaluate_all_models(
    dataset_csv_path: Path = None,
    sampling_rate: float = 50.0
) -> Dict[str, Any]:
    """
    Complete end-to-end model training, benchmarking, and selection pipeline.
    """
    logger.info("=" * 70)
    logger.info("  FALLGUARD AI - MULTI-MODEL TRAINING & BENCHMARKING PIPELINE")
    logger.info("=" * 70)
    
    if dataset_csv_path is None:
        dataset_csv_path = PROCESSED_DIR / "unified_sensor_stream.csv"
        
    if not dataset_csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_csv_path}. Run data/download_datasets.py first!")
        
    logger.info(f"Loading raw sensor stream from {dataset_csv_path}...")
    df_raw = pd.read_csv(dataset_csv_path)
    df_clean = clean_sensor_stream(df_raw)
    
    # 1. Feature Extraction (with persistent cache for fast iterations)
    features_csv = PROCESSED_DIR / "extracted_features.csv"
    if features_csv.exists():
        logger.info(f"Loading cached window features from {features_csv}...")
        df_features = pd.read_csv(features_csv)
    else:
        logger.info("Extracting time-domain, dynamic, and frequency features from sliding windows...")
        df_features = extract_dataset_features(df_clean, sampling_rate=sampling_rate)
        df_features.to_csv(features_csv, index=False)
        logger.info(f"Saved extracted features cache to {features_csv}")
        
    logger.info(f"Loaded features for {len(df_features)} total windows across {df_features['activity'].nunique()} activities.")
    
    # Save activity distribution plot
    plot_activity_distribution(df_features, FIGURES_DIR / "activity_distribution.png")
    
    # 2. Subject-Independent Split (Strict zero data leakage)
    logger.info("Partitioning data with subject-independent train/test splitting...")
    train_df, test_df, train_subjects, test_subjects = split_by_subject(df_features, test_subject_ratio=0.30, random_seed=42)
    
    # Non-feature metadata columns
    meta_cols = ["window_id", "subject_id", "activity", "is_fall", "split_origin"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    
    logger.info(f"Identified {len(feature_cols)} numeric features.")
    
    # Extract matrices
    X_train_raw = train_df[feature_cols].to_numpy()
    X_test_raw = test_df[feature_cols].to_numpy()
    
    # Handle any potential NaNs in features
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode targets
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["activity"])
    y_test = label_encoder.transform(test_df["activity"])
    class_names = list(label_encoder.classes_)
    
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Train samples: {len(X_train_raw)} (Subjects: {train_subjects})")
    logger.info(f"Test samples: {len(X_test_raw)} (Subjects: {test_subjects})")
    
    # Fit StandardScaler ONLY on Training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 3. Define 5 Candidate ML Models
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "use_scaled": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=12, min_samples_split=4, class_weight="balanced", random_state=42),
            "use_scaled": False
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_split=4, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
            "use_scaled": False
        },
        "Support Vector Machine": {
            "model": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=42),
            "use_scaled": True
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            "use_scaled": False
        }
    }
    
    results = {}
    comparison_rows = []
    
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<24} | {'Accuracy':<9} | {'Macro F1':<9} | {'Fall Recall':<11} | {'Fall Prec':<10} | {'Score':<8}")
    print("-" * 80)
    
    best_model_name = None
    best_composite_score = -1.0
    best_eval = None
    best_trained_model = None
    best_use_scaled = False
    
    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        
        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_te = X_test_scaled if use_scaled else X_test_raw
        
        logger.info(f"Training {name}...")
        clf.fit(X_tr, y_train)
        
        eval_metrics = evaluate_classifier(clf, X_te, y_test, class_names, fall_class_name="FALL")
        results[name] = eval_metrics
        
        acc = eval_metrics["accuracy"]
        f1 = eval_metrics["macro_f1"]
        fall_rec = eval_metrics["fall_recall"]
        fall_prec = eval_metrics["fall_precision"]
        
        # Composite score: 50% Fall Recall, 30% Macro F1, 20% Accuracy
        composite_score = 0.50 * fall_rec + 0.30 * f1 + 0.20 * acc
        
        print(f"{name:<24} | {acc*100:6.2f}%   | {f1*100:6.2f}%   | {fall_rec*100:7.2f}%    | {fall_prec*100:6.2f}%   | {composite_score*100:5.2f}%")
        
        comparison_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Macro F1": f1,
            "Fall Recall": fall_rec,
            "Fall Precision": fall_prec,
            "Weighted F1": eval_metrics["weighted_f1"],
            "Fall FPR": eval_metrics["fall_fpr"],
            "Fall FNR": eval_metrics["fall_fnr"],
            "Fall ROC-AUC": eval_metrics["fall_roc_auc"],
            "Composite Score": composite_score
        })
        
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_model_name = name
            best_eval = eval_metrics
            best_trained_model = clf
            best_use_scaled = use_scaled
            
    print("=" * 80)
    print(f"\n[*] BEST SELECTED MODEL: {best_model_name} (Composite Score: {best_composite_score*100:.2f}%)\n")
    
    # 4. Generate Visualizations
    df_comparison = pd.DataFrame(comparison_rows)
    plot_model_comparison(df_comparison, FIGURES_DIR / "model_comparison_bar.png")
    
    # Confusion Matrix for Best Model
    plot_confusion_matrix(best_eval["confusion_matrix"], class_names, best_model_name, FIGURES_DIR / "confusion_matrix_best.png")
    
    # Feature Importance (for tree models or permutation)
    feat_importance = {}
    if hasattr(best_trained_model, "feature_importances_"):
        importances = best_trained_model.feature_importances_
        feat_importance = {feat: float(imp) for feat, imp in zip(feature_cols, importances)}
    elif hasattr(best_trained_model, "coef_"):
        # Magnitude of coefficients
        importances = np.mean(np.abs(best_trained_model.coef_), axis=0)
        feat_importance = {feat: float(imp) for feat, imp in zip(feature_cols, importances)}
        
    plot_feature_importance(feat_importance, top_n=20, save_path=FIGURES_DIR / "feature_importance_top20.png")
    
    # 5. Serialize Artifact Bundle
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    fall_idx = class_to_idx.get("FALL", -1)
    
    model_bundle = {
        "model": best_trained_model,
        "scaler": scaler,
        "use_scaled": best_use_scaled,
        "feature_names": feature_cols,
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "fall_class_idx": fall_idx,
        "model_name": best_model_name,
        "metrics": {
            "accuracy": best_eval["accuracy"],
            "macro_precision": best_eval["macro_precision"],
            "macro_recall": best_eval["macro_recall"],
            "macro_f1": best_eval["macro_f1"],
            "weighted_f1": best_eval["weighted_f1"],
            "fall_recall": best_eval["fall_recall"],
            "fall_precision": best_eval["fall_precision"],
            "fall_f1": best_eval["fall_f1"],
            "fall_fpr": best_eval["fall_fpr"],
            "fall_fnr": best_eval["fall_fnr"],
            "fall_roc_auc": best_eval["fall_roc_auc"],
            "confusion_matrix": best_eval["confusion_matrix"]
        },
        "feature_importance": feat_importance,
        "trained_at": datetime.utcnow().isoformat(),
        "sampling_rate": sampling_rate,
        "window_size_seconds": 2.56,
        "window_samples": int(sampling_rate * 2.56),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "total_train_windows": len(X_train_raw),
        "total_test_windows": len(X_test_raw)
    }
    
    joblib.dump(model_bundle, FINAL_MODEL_PATH)
    logger.info(f"Successfully saved final model bundle to {FINAL_MODEL_PATH}")
    
    # Save metrics JSON
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(model_bundle["metrics"], f, indent=2)
        
    with open(COMPARISON_JSON_PATH, "w") as f:
        json.dump(comparison_rows, f, indent=2)
        
    logger.info(f"Saved evaluation metrics to {METRICS_JSON_PATH}")
    return model_bundle
