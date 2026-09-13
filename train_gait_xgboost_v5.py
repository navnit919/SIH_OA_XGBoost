
# ============================================================
# OA GAIT XGBOOST V5
# Subject-wise cross-validation
# Final corrected version
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
# CONFIGURATION
# ============================================================

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

THRESHOLD = 0.50

RANDOM_STATE = 42

# ============================================================
# XGBOOST PARAMETERS
# ============================================================

XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 2,
    "learning_rate": 0.035,
    "min_child_weight": 5,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ============================================================
# START
# ============================================================

print("=" * 70)
print("OA GAIT XGBOOST V5 TRAINING")
print("=" * 70)

# ============================================================
# CHECK DATA FILE
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nDataset: {DATA_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# VERIFY DATASET SIZE
# ============================================================

if len(df) != 188:
    print(
        f"\nWARNING: Expected 188 trials, "
        f"but found {len(df)}."
    )

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET, GROUP]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")

    for col in missing_columns:
        print("  -", col)

    raise SystemExit(1)

# ============================================================
# CLEAN DATA
# ============================================================

before_rows = len(df)

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

after_rows = len(df)

if before_rows != after_rows:
    print(
        f"\nWARNING: Removed "
        f"{before_rows - after_rows} rows containing missing values."
    )

# ============================================================
# PREPARE X / Y / GROUPS
# ============================================================

X = df[FEATURES].astype(float)

y = df[TARGET].astype(int)

groups = df[GROUP].astype(str)

# ============================================================
# DATASET SUMMARY
# ============================================================

print("\nFeatures:")

for i, feature in enumerate(FEATURES, 1):
    print(f"{i:02d}. {feature}")

print("\nClass distribution:")

print(
    y.value_counts()
    .sort_index()
)

print(
    f"\nSubjects: {groups.nunique()}"
)

negative = int(
    (y == 0).sum()
)

positive = int(
    (y == 1).sum()
)

print(
    f"Non-OA: {negative}"
)

print(
    f"OA:     {positive}"
)

if positive == 0:
    raise SystemExit(
        "ERROR: No OA samples found."
    )

scale_pos_weight = negative / positive

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.4f}"
)

# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_probability = np.zeros(
    len(df),
    dtype=float
)

oof_prediction = np.zeros(
    len(df),
    dtype=int
)

fold_results = []

# ============================================================
# CROSS-VALIDATION START
# ============================================================

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

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # STORE OOF PREDICTIONS
    # --------------------------------------------------------

    oof_probability[test_idx] = probability

    oof_prediction[test_idx] = prediction

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm_fold = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm_fold.ravel()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        prediction
    )

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
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
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

        auc = float("nan")

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    fold_results.append(
        {
            "fold": int(fold),
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_subjects": int(
                train_groups.nunique()
            ),
            "test_subjects": int(
                test_groups.nunique()
            ),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(f1),
            "roc_auc": (
                None
                if np.isnan(auc)
                else float(auc)
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    )

    # --------------------------------------------------------
    # PRINT FOLD RESULTS
    # --------------------------------------------------------

    print(f"\nFold {fold}")
    print("-" * 40)

    print(
        f"Train samples : {len(train_idx)}"
    )

    print(
        f"Test samples  : {len(test_idx)}"
    )

    print(
        f"Train subjects: {train_groups.nunique()}"
    )

    print(
        f"Test subjects : {test_groups.nunique()}"
    )

    print(
        f"Accuracy      : {accuracy:.4f}"
    )

    print(
        f"Precision     : {precision:.4f}"
    )

    print(
        f"Sensitivity   : {sensitivity:.4f}"
    )

    print(
        f"Specificity   : {specificity:.4f}"
    )

    print(
        f"F1            : {f1:.4f}"
    )

    print(
        f"ROC-AUC       : {auc:.4f}"
    )

# ============================================================
# OVERALL OOF PERFORMANCE
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

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

f1 = f1_score(
    y,
    oof_prediction,
    zero_division=0
)

auc = roc_auc_score(
    y,
    oof_probability
)

# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print(
    f"\nAccuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Sensitivity : {sensitivity:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"F1 Score    : {f1:.4f}"
)

print(
    f"ROC-AUC     : {auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print(
    f"\nTN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

final_model = XGBClassifier(
    **XGB_PARAMS,
    scale_pos_weight=scale_pos_weight
)

final_model.fit(
    X,
    y
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    final_model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

print("\nFeature Importance")
print("-" * 40)

for feature, value in importance.items():

    print(
        f"{feature:25s} "
        f"{value:.6f}"
    )

# ============================================================
# MODEL PACKAGE
# ============================================================
#
# IMPORTANT:
# The XGBClassifier is saved ONLY inside the .pkl file.
#
# The JSON file contains only serializable information.
# ============================================================

model_package = {
    "model": final_model,

    "features": FEATURES,

    "threshold": THRESHOLD,

    "version": "v5",

    "dataset": {
        "rows": int(len(df)),
        "subjects": int(groups.nunique()),
        "oa_trials": int(positive),
        "non_oa_trials": int(negative)
    },

    "validation": {
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": RANDOM_STATE
    },

    "xgboost_parameters": {
        key: (
            float(value)
            if isinstance(
                value,
                (np.floating, float)
            )
            else int(value)
            if isinstance(
                value,
                (np.integer, int)
            )
            else value
        )
        for key, value in {
            **XGB_PARAMS,
            "scale_pos_weight": scale_pos_weight
        }.items()
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
        str(feature): float(value)
        for feature, value in importance.items()
    },

    "fold_results": fold_results
}

# ============================================================
# SAVE PKL MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model_package,
    MODEL_PATH
)

print(
    "Model saved successfully:"
)

print(
    MODEL_PATH
)

# ============================================================
# JSON METRICS PACKAGE
# ============================================================
#
# DO NOT PUT model_package directly into json.dump()
# because it contains XGBClassifier.
# ============================================================

metrics_package = {
    "version": "v5",

    "dataset": {
        "rows": int(len(df)),
        "subjects": int(groups.nunique()),
        "oa_trials": int(positive),
        "non_oa_trials": int(negative)
    },

    "features": FEATURES,

    "threshold": float(THRESHOLD),

    "validation": {
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": RANDOM_STATE
    },

    "xgboost_parameters": {
        key: (
            float(value)
            if isinstance(
                value,
                (np.floating, float)
            )
            else int(value)
            if isinstance(
                value,
                (np.integer, int)
            )
            else value
        )
        for key, value in {
            **XGB_PARAMS,
            "scale_pos_weight": scale_pos_weight
        }.items()
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
        str(feature): float(value)
        for feature, value in importance.items()
    },

    "fold_results": fold_results
}

# ============================================================
# SAVE JSON
# ============================================================

print("\nSaving metrics...")

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

print(
    "Metrics saved successfully:"
)

print(
    METRICS_PATH
)

# ============================================================
# VERIFY FILES
# ============================================================

model_exists = os.path.exists(
    MODEL_PATH
)

metrics_exists = os.path.exists(
    METRICS_PATH
)

print("\n" + "=" * 70)
print("FILE VERIFICATION")
print("=" * 70)

print(
    f"Model file exists   : {model_exists}"
)

print(
    f"Metrics file exists : {metrics_exists}"
)

if model_exists:

    print(
        f"Model size         : "
        f"{os.path.getsize(MODEL_PATH):,} bytes"
    )

if metrics_exists:

    print(
        f"Metrics size       : "
        f"{os.path.getsize(METRICS_PATH):,} bytes"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V5 TRAINING COMPLETE")
print("=" * 70)

print("\nDataset")
print("-" * 40)

print(
    f"Trials       : {len(df)}"
)

print(
    f"Subjects     : {groups.nunique()}"
)

print(
    f"OA trials    : {positive}"
)

print(
    f"Non-OA trials: {negative}"
)

print("\nFinal V5 OOF Results")
print("-" * 40)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Sensitivity : {sensitivity * 100:.2f}%"
)

print(
    f"Specificity : {specificity * 100:.2f}%"
)

print(
    f"F1 Score    : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC     : {auc * 100:.2f}%"
)

print("\nConfusion Matrix")
print(cm)

print("\nModel:")
print(MODEL_PATH)

print("\nMetrics:")
print(METRICS_PATH)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)