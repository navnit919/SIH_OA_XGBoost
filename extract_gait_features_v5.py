
# ============================================================
# OA GAIT XGBOOST V5
# Subject-wise cross-validation
# ============================================================

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
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\osteoarthrits\SIH_OA_XGBoost"

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "gait",
    "gait_marker_cohort_v5.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gait"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "oa_xgboost_gait_model_v5.pkl"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "oa_xgboost_gait_metrics_v5.json"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("OA GAIT XGBOOST V5 TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset: {DATA_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# FEATURES
# ============================================================
# Controlled V5 feature set.
#
# Six original knee features + four gait dynamics features.
# This avoids using too many features for only 27 subjects.

FEATURES = [
    "knee_rom",
    "knee_mean",
    "knee_std",
    "knee_max_flex",
    "knee_min_flex",
    "knee_median",

    "mean_velocity",
    "velocity_std",
    "rom_std",
    "max_flex_timing"
]

TARGET = "label"
GROUP = "subject"

# ============================================================
# CHECK COLUMNS
# ============================================================

missing_features = [
    f for f in FEATURES
    if f not in df.columns
]

if missing_features:
    print("\nERROR: Missing feature columns:")
    for f in missing_features:
        print("  -", f)

    raise SystemExit(1)

if TARGET not in df.columns:
    raise SystemExit(
        f"ERROR: Target column '{TARGET}' not found."
    )

if GROUP not in df.columns:
    raise SystemExit(
        f"ERROR: Group column '{GROUP}' not found."
    )

# ============================================================
# DATA CLEANING
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET, GROUP]
).reset_index(drop=True)

X = df[FEATURES].astype(float)
y = df[TARGET].astype(int)
groups = df[GROUP].astype(str)

print("\nFeature set:")
for i, feature in enumerate(FEATURES, 1):
    print(f"  {i:02d}. {feature}")

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nSubject count:", groups.nunique())

# ============================================================
# CLASS WEIGHT
# ============================================================

negative = int((y == 0).sum())
positive = int((y == 1).sum())

scale_pos_weight = negative / positive

print(f"\nNon-OA samples: {negative}")
print(f"OA samples:     {positive}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probability = np.zeros(len(df))
oof_prediction = np.zeros(len(df), dtype=int)

fold_results = []

print("\n" + "=" * 70)
print("5-FOLD SUBJECT-WISE CROSS-VALIDATION")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.035,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(X_test)[:, 1]

    # Fixed threshold for unbiased comparison
    prediction = (probability >= 0.50).astype(int)

    oof_probability[test_idx] = probability
    oof_prediction[test_idx] = prediction

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    sensitivity = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    specificity = (
        confusion_matrix(
            y_test,
            prediction,
            labels=[0, 1]
        )[0, 0]
        /
        max(
            confusion_matrix(
                y_test,
                prediction,
                labels=[0, 1]
            )[0].sum(),
            1
        )
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            y_test,
            probability
        )
    except ValueError:
        auc = np.nan

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "train_subjects": groups.iloc[train_idx].nunique(),
        "test_subjects": groups.iloc[test_idx].nunique()
    })

    print(f"\nFold {fold}")
    print("-" * 40)
    print(f"Train samples : {len(train_idx)}")
    print(f"Test samples  : {len(test_idx)}")
    print(f"Train subjects: {groups.iloc[train_idx].nunique()}")
    print(f"Test subjects : {groups.iloc[test_idx].nunique()}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Sensitivity   : {sensitivity:.4f}")
    print(f"Specificity   : {specificity:.4f}")
    print(f"F1            : {f1:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")

# ============================================================
# OVERALL OOF METRICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 70)

accuracy = accuracy_score(
    y,
    oof_prediction
)

precision = precision_score(
    y,
    oof_prediction,
    zero_division=0
)

sensitivity = recall_score(
    y,
    oof_prediction,
    zero_division=0
)

cm = confusion_matrix(
    y,
    oof_prediction,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / max(tn + fp, 1)

f1 = f1_score(
    y,
    oof_prediction,
    zero_division=0
)

auc = roc_auc_score(
    y,
    oof_probability
)

print(f"\nAccuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Train final model on all available data
final_model = XGBClassifier(
    n_estimators=250,
    max_depth=2,
    learning_rate=0.035,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.5,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

importance = pd.Series(
    final_model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

for feature, value in importance.items():
    print(f"{feature:25s} {value:.6f}")

# ============================================================
# SAVE MODEL PACKAGE
# ============================================================

model_package = {
    "model": final_model,
    "features": FEATURES,
    "threshold": 0.50,

    "version": "v5",

    "dataset": {
        "rows": int(len(df)),
        "subjects": int(groups.nunique()),
        "oa_trials": int(positive),
        "non_oa_trials": int(negative)
    },

    "feature_definition": {
        "knee_rom": "max flexion - min flexion",
        "knee_mean": "mean knee flexion",
        "knee_std": "standard deviation of knee flexion",
        "knee_max_flex": "maximum knee flexion",
        "knee_min_flex": "minimum knee flexion",
        "knee_median": "median knee flexion",
        "mean_velocity": "mean absolute knee angular velocity",
        "velocity_std": "standard deviation of knee angular velocity",
        "rom_std": "standard deviation of cycle/segment ROM",
        "max_flex_timing": "normalized timing of maximum flexion"
    },

    "validation": {
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42
    },

    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc)
    },

    "confusion_matrix": cm.tolist(),

    "feature_importance": {
        str(k): float(v)
        for k, v in importance.items()
    },

    "fold_results": fold_results
}

joblib.dump(
    model_package,
    MODEL_PATH
)

# ============================================================
# SAVE METRICS JSON
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_package,
        f,
        indent=4
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V5 TRAINING COMPLETE")
print("=" * 70)

print("\nModel saved:")
print(MODEL_PATH)

print("\nMetrics saved:")
print(METRICS_PATH)

print("\nFinal feature count:", len(FEATURES))

print("\nV5 OOF RESULTS")
print("-" * 40)
print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Sensitivity : {sensitivity * 100:.2f}%")
print(f"Specificity : {specificity * 100:.2f}%")
print(f"F1 Score    : {f1 * 100:.2f}%")
print(f"ROC-AUC     : {auc * 100:.2f}%")

print("\nDone.")
