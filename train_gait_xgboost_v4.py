
import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\osteoarthrits\SIH_OA_XGBoost"

DATASET = os.path.join(
    BASE_DIR,
    "data",
    "gait",
    "gait_marker_cohort_right_leg.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gait"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "oa_xgboost_gait_model_v4.pkl"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "oa_xgboost_gait_metrics_v4.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# FEATURES
# IMPORTANT:
# These MUST remain identical to camera/dashboard features.
# ============================================================

FEATURES = [
    "knee_rom",
    "knee_mean",
    "knee_std",
    "knee_max_flex",
    "knee_min_flex",
    "knee_median",
]

TARGET = "label"
GROUP_COLUMN = "subject"

# ============================================================
# MODEL PARAMETERS
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "min_child_weight": 2,
    "gamma": 0.10,
    "reg_alpha": 0.20,
    "reg_lambda": 2.0,

    # Handle class imbalance
    "scale_pos_weight": 127 / 61,

    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# ============================================================
# THRESHOLDS TO TEST
# ============================================================

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OA GAIT XGBOOST V4 TRAINING")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING GAIT DATASET")
print("=" * 70)

print(f"Dataset: {DATASET}")

df = pd.read_csv(DATASET)

print(f"Shape: {df.shape}")

# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = FEATURES + [TARGET, GROUP_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[FEATURES].isnull().any().any():
    raise ValueError(
        "Dataset contains missing values in gait features."
    )

# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Trials   : {len(df)}")
print(f"Subjects : {df[GROUP_COLUMN].nunique()}")

print("\nLabels:")
print(df[TARGET].value_counts())

if "group" in df.columns:
    print("\nGroups:")
    print(df["group"].value_counts())

print("\nTrials per subject:")
print(df.groupby(GROUP_COLUMN).size().describe())

# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()
groups = df[GROUP_COLUMN].copy()

# ============================================================
# SUBJECT-WISE CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = np.zeros(len(df))
fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print("\n" + "-" * 70)
    print(f"FOLD {fold}")
    print("-" * 70)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print(f"Train trials   : {len(train_idx)}")
    print(f"Test trials    : {len(test_idx)}")
    print(f"Train subjects : {train_groups.nunique()}")
    print(f"Test subjects  : {test_groups.nunique()}")

    print(
        "Test subjects  : "
        + ", ".join(sorted(test_groups.unique()))
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    overlap = set(train_groups) & set(test_groups)

    if overlap:
        raise RuntimeError(
            f"SUBJECT LEAKAGE DETECTED: {overlap}"
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = XGBClassifier(**MODEL_PARAMS)

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PROBABILITY PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

    oof_probabilities[test_idx] = probabilities

    # --------------------------------------------------------
    # DEFAULT THRESHOLD = 0.50
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    try:
        auc = roc_auc_score(
            y_test,
            probabilities
        )
    except ValueError:
        auc = float("nan")

    print(f"\nAccuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Sensitivity    : {sensitivity:.4f}")
    print(f"Specificity    : {specificity:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"ROC-AUC        : {auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    fold_results.append({
        "fold": fold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "train_trials": len(train_idx),
        "test_trials": len(test_idx),
        "train_subjects": int(train_groups.nunique()),
        "test_subjects": int(test_groups.nunique()),
    })

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL OUT-OF-FOLD THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []

for threshold in THRESHOLDS:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y,
        predictions,
        zero_division=0
    )

    specificity = recall_score(
        y,
        predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    })

    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy {accuracy:.4f} | "
        f"Precision {precision:.4f} | "
        f"Sensitivity {sensitivity:.4f} | "
        f"Specificity {specificity:.4f} | "
        f"F1 {f1:.4f}"
    )

# ============================================================
# SELECT THRESHOLD
# ============================================================

# For screening, prioritize sensitivity while
# maintaining reasonable specificity.

valid_thresholds = [
    r for r in threshold_results
    if r["specificity"] >= 0.50
]

if valid_thresholds:

    best_threshold_result = max(
        valid_thresholds,
        key=lambda r: (
            r["sensitivity"],
            r["f1"]
        )
    )

else:

    best_threshold_result = max(
        threshold_results,
        key=lambda r: r["f1"]
    )

BEST_THRESHOLD = best_threshold_result["threshold"]

print("\n" + "=" * 70)
print("SELECTED DECISION THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {BEST_THRESHOLD:.2f}"
)

print(
    f"Sensitivity : "
    f"{best_threshold_result['sensitivity']:.4f}"
)

print(
    f"Specificity : "
    f"{best_threshold_result['specificity']:.4f}"
)

print(
    f"F1-score    : "
    f"{best_threshold_result['f1']:.4f}"
)

# ============================================================
# OVERALL OOF PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 70)

final_oof_predictions = (
    oof_probabilities >= BEST_THRESHOLD
).astype(int)

accuracy = accuracy_score(
    y,
    final_oof_predictions
)

precision = precision_score(
    y,
    final_oof_predictions,
    zero_division=0
)

sensitivity = recall_score(
    y,
    final_oof_predictions,
    zero_division=0
)

specificity = recall_score(
    y,
    final_oof_predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    y,
    final_oof_predictions,
    zero_division=0
)

auc = roc_auc_score(
    y,
    oof_probabilities
)

cm = confusion_matrix(
    y,
    final_oof_predictions
)

print(f"Threshold   : {BEST_THRESHOLD:.2f}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

print("\nConfusion matrix:")
print(cm)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Train temporary model on complete dataset
importance_model = XGBClassifier(**MODEL_PARAMS)

importance_model.fit(
    X,
    y
)

feature_importance = (
    importance_model.feature_importances_
)

for feature, importance in sorted(
    zip(FEATURES, feature_importance),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{feature:20s}: "
        f"{importance:.6f}"
    )

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 70)

final_model = XGBClassifier(**MODEL_PARAMS)

final_model.fit(
    X,
    y
)

# ============================================================
# SAVE MODEL PACKAGE
# ============================================================

model_package = {
    "model": final_model,

    "features": FEATURES,

    "feature_order": FEATURES,

    "target": TARGET,

    "group_column": GROUP_COLUMN,

    "threshold": float(BEST_THRESHOLD),

    "angle_definition":
        "knee flexion = 180 - internal 3D knee angle",

    "hip_definition":
        "midpoint(LASIS,RASIS), fallback RASIS",

    "knee_marker":
        "MKNEE",

    "ankle_marker":
        "MANK",

    "training_trials":
        int(len(df)),

    "training_subjects":
        int(df[GROUP_COLUMN].nunique()),

    "model_version":
        "oa_xgboost_gait_model_v4",
}

joblib.dump(
    model_package,
    MODEL_PATH
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics_package = {

    "dataset": DATASET,

    "training_trials": int(len(df)),

    "training_subjects":
        int(df[GROUP_COLUMN].nunique()),

    "label_distribution":
        {
            str(k): int(v)
            for k, v in y.value_counts().items()
        },

    "features": FEATURES,

    "fold_results": fold_results,

    "threshold_results": threshold_results,

    "selected_threshold":
        float(BEST_THRESHOLD),

    "overall_oof": {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "roc_auc":
            float(auc),

        "confusion_matrix":
            cm.tolist(),
    },

    "model_parameters":
        MODEL_PARAMS,
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_package,
        f,
        indent=4
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nMetrics:")
print(METRICS_PATH)

print("\nSelected threshold:")
print(f"{BEST_THRESHOLD:.2f}")

print("\nFeatures:")

for feature in FEATURES:
    print(f"  - {feature}")

print("\nThe v4 model is ready for camera/dashboard integration.")
print("=" * 70)
