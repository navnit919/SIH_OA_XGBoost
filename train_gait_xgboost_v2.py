import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

DATA = r"D:\osteoarthrits\SIH_OA_XGBoost\gait_marker_cohort_right_leg.csv"

MODEL_OUT = r"D:\osteoarthrits\SIH_OA_XGBoost\oa_xgboost_gait_model_v2.pkl"

META_OUT = r"D:\osteoarthrits\SIH_OA_XGBoost\oa_xgboost_gait_model_v2_metadata.json"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)

print("=" * 80)
print("OA GAIT XGBOOST TRAINING")
print("=" * 80)

print("\nDataset:", DATA)
print("Shape:", df.shape)


# ============================================================
# FEATURES
# ============================================================

COMPACT_FEATURES = [
    "knee_rom",
    "knee_mean",
    "knee_std",
    "knee_min_flex"
]

FULL_FEATURES = [
    "knee_rom",
    "knee_mean",
    "knee_std",
    "knee_max_flex",
    "knee_min_flex",
    "knee_median"
]


# ============================================================
# DATA
# ============================================================

y = df["label"].astype(int)
groups = df["subject"].astype(str)


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model():

    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )


# ============================================================
# SUBJECT-LEVEL CV
# ============================================================

def evaluate_features(feature_list, name):

    print("\n")
    print("=" * 80)
    print("MODEL:", name)
    print("=" * 80)

    X = df[feature_list].copy()

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    trial_predictions = []
    trial_truth = []
    trial_subjects = []

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        subjects_train = groups.iloc[train_idx]
        subjects_test = groups.iloc[test_idx]

        print(f"\nFold {fold}")

        print(
            "  Train subjects:",
            subjects_train.nunique()
        )

        print(
            "  Test subjects :",
            subjects_test.nunique()
        )

        # Check leakage
        overlap = set(
            subjects_train
        ).intersection(
            set(subjects_test)
        )

        if overlap:

            raise RuntimeError(
                f"SUBJECT LEAKAGE DETECTED: {overlap}"
            )

        model = make_model()

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        acc = accuracy_score(
            y_test,
            predictions
        )

        bal_acc = balanced_accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:

            auc = roc_auc_score(
                y_test,
                probabilities
            )

        except ValueError:

            auc = np.nan

        print(
            f"  Accuracy          : {acc:.3f}"
        )

        print(
            f"  Balanced Accuracy : {bal_acc:.3f}"
        )

        print(
            f"  Precision         : {precision:.3f}"
        )

        print(
            f"  Recall            : {recall:.3f}"
        )

        print(
            f"  F1                : {f1:.3f}"
        )

        print(
            f"  ROC-AUC           : {auc:.3f}"
        )

        print(
            "  Confusion Matrix:"
        )

        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        fold_results.append({
            "fold": fold,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc
        })

        trial_predictions.extend(
            probabilities.tolist()
        )

        trial_truth.extend(
            y_test.tolist()
        )

        trial_subjects.extend(
            subjects_test.tolist()
        )


    # ========================================================
    # TRIAL-LEVEL CV SUMMARY
    # ========================================================

    trial_predictions = np.array(
        trial_predictions
    )

    trial_truth = np.array(
        trial_truth
    )

    trial_classes = (
        trial_predictions >= 0.5
    ).astype(int)

    print("\n")
    print("-" * 80)
    print("TRIAL-LEVEL CV SUMMARY")
    print("-" * 80)

    print(
        "Accuracy:",
        f"{accuracy_score(trial_truth, trial_classes):.3f}"
    )

    print(
        "Balanced accuracy:",
        f"{balanced_accuracy_score(trial_truth, trial_classes):.3f}"
    )

    print(
        "Precision:",
        f"{precision_score(trial_truth, trial_classes, zero_division=0):.3f}"
    )

    print(
        "Recall:",
        f"{recall_score(trial_truth, trial_classes, zero_division=0):.3f}"
    )

    print(
        "F1:",
        f"{f1_score(trial_truth, trial_classes, zero_division=0):.3f}"
    )

    try:

        trial_auc = roc_auc_score(
            trial_truth,
            trial_predictions
        )

        print(
            "ROC-AUC:",
            f"{trial_auc:.3f}"
        )

    except ValueError:

        trial_auc = np.nan


    # ========================================================
    # SUBJECT-LEVEL AGGREGATION
    # ========================================================

    pred_df = pd.DataFrame({
        "subject": trial_subjects,
        "truth": trial_truth,
        "probability": trial_predictions
    })

    subject_df = (
        pred_df
        .groupby("subject")
        .agg(
            truth=("truth", "first"),
            probability=("probability", "mean"),
            n_trials=("probability", "size")
        )
        .reset_index()
    )

    subject_df["prediction"] = (
        subject_df["probability"] >= 0.5
    ).astype(int)

    print("\n")
    print("-" * 80)
    print("SUBJECT-LEVEL CV SUMMARY")
    print("-" * 80)

    print(
        "Subjects evaluated:",
        len(subject_df)
    )

    subject_acc = accuracy_score(
        subject_df["truth"],
        subject_df["prediction"]
    )

    subject_bal_acc = balanced_accuracy_score(
        subject_df["truth"],
        subject_df["prediction"]
    )

    subject_precision = precision_score(
        subject_df["truth"],
        subject_df["prediction"],
        zero_division=0
    )

    subject_recall = recall_score(
        subject_df["truth"],
        subject_df["prediction"],
        zero_division=0
    )

    subject_f1 = f1_score(
        subject_df["truth"],
        subject_df["prediction"],
        zero_division=0
    )

    try:

        subject_auc = roc_auc_score(
            subject_df["truth"],
            subject_df["probability"]
        )

    except ValueError:

        subject_auc = np.nan


    print(
        "Accuracy:",
        f"{subject_acc:.3f}"
    )

    print(
        "Balanced accuracy:",
        f"{subject_bal_acc:.3f}"
    )

    print(
        "Precision:",
        f"{subject_precision:.3f}"
    )

    print(
        "Recall:",
        f"{subject_recall:.3f}"
    )

    print(
        "F1:",
        f"{subject_f1:.3f}"
    )

    print(
        "ROC-AUC:",
        f"{subject_auc:.3f}"
    )

    print("\nSubject predictions:")

    display_df = subject_df.copy()

    display_df["probability"] = (
        display_df["probability"]
        .round(3)
    )

    print(
        display_df.to_string(
            index=False
        )
    )


    return {
        "name": name,
        "features": feature_list,
        "trial_accuracy": accuracy_score(
            trial_truth,
            trial_classes
        ),
        "trial_balanced_accuracy": balanced_accuracy_score(
            trial_truth,
            trial_classes
        ),
        "trial_precision": precision_score(
            trial_truth,
            trial_classes,
            zero_division=0
        ),
        "trial_recall": recall_score(
            trial_truth,
            trial_classes,
            zero_division=0
        ),
        "trial_f1": f1_score(
            trial_truth,
            trial_classes,
            zero_division=0
        ),
        "trial_roc_auc": trial_auc,
        "subject_accuracy": subject_acc,
        "subject_balanced_accuracy": subject_bal_acc,
        "subject_precision": subject_precision,
        "subject_recall": subject_recall,
        "subject_f1": subject_f1,
        "subject_roc_auc": subject_auc
    }


# ============================================================
# RUN BOTH EXPERIMENTS
# ============================================================

compact_results = evaluate_features(
    COMPACT_FEATURES,
    "COMPACT"
)

full_results = evaluate_features(
    FULL_FEATURES,
    "FULL"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame([
    compact_results,
    full_results
])

print(
    comparison[
        [
            "name",
            "subject_accuracy",
            "subject_balanced_accuracy",
            "subject_precision",
            "subject_recall",
            "subject_f1",
            "subject_roc_auc"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# SELECT MODEL
#
# Prefer subject-level balanced accuracy/F1 rather than
# raw trial accuracy because the dataset has repeated trials.
# ============================================================

best = max(
    [compact_results, full_results],
    key=lambda r: (
        r["subject_balanced_accuracy"],
        r["subject_f1"]
    )
)

print("\n")
print("=" * 80)
print("SELECTED MODEL")
print("=" * 80)

print("Model:", best["name"])

print(
    "Features:",
    best["features"]
)


# ============================================================
# TRAIN FINAL MODEL ON ALL TRIALS
#
# This is done ONLY after the grouped CV experiment.
# ============================================================

final_features = best["features"]

X_final = df[final_features]
y_final = df["label"]

final_model = make_model()

final_model.fit(
    X_final,
    y_final
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    MODEL_OUT
)


metadata = {
    "model_type": "XGBClassifier",
    "task": "binary OA screening",
    "positive_class": "OA",
    "negative_classes": ["Y", "OH"],
    "feature_set": final_features,
    "dataset": os.path.basename(DATA),
    "n_trials": int(len(df)),
    "n_subjects": int(df["subject"].nunique()),
    "oa_subjects": int(
        df.loc[df["label"] == 1, "subject"].nunique()
    ),
    "non_oa_subjects": int(
        df.loc[df["label"] == 0, "subject"].nunique()
    ),
    "validation": "StratifiedGroupKFold, groups=subject",
    "random_state": 42,
    "cv_results": {
        "compact": compact_results,
        "full": full_results
    }
}

with open(
    META_OUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL SAVED")
print("=" * 80)

print(
    "Model:",
    MODEL_OUT
)

print(
    "Metadata:",
    META_OUT
)

print(
    "\nIMPORTANT:"
)

print(
    "This model is an experimental OA screening/risk model."
)

print(
    "It is NOT a clinical diagnostic model."
)

print(
    "\nDONE"
)
